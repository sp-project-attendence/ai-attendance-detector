"""Run tracking, face sampling, identity voting, and M2 exports for one video."""
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from faces import detect_faces, embed, make_models


def load_config(path):
    """Read the small flat key:value subset used by config.yaml (no new package)."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw = (part.strip() for part in line.split(":", 1))
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = float(raw) if any(c in raw for c in ".eE") else int(raw)
            except ValueError:
                value = raw.strip("\"'")
        values[key] = value
    return values


def load_gallery(database):
    """Load all consented per-photo embeddings, grouped by student_id."""
    gallery = defaultdict(list)
    with sqlite3.connect(database) as connection:
        rows = connection.execute("""
            SELECT s.student_id, e.embedding
            FROM students s JOIN enrolment_embeddings e USING(student_id)
            WHERE lower(trim(s.consent))='yes'
            ORDER BY s.student_id, e.embedding_id
        """).fetchall()
    for student_id, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if vector.size and np.isfinite(vector).all():
            norm = float(np.linalg.norm(vector))
            if norm:
                gallery[student_id].append(vector / norm)
    if not gallery:
        raise ValueError("No consented enrolment embeddings found in the SQLite database")
    return {sid: np.vstack(vectors) for sid, vectors in gallery.items()}


def quality_weight(sample):
    """Bounded weight based on face detector score and crop sharpness."""
    sharp_factor = min(1.0, float(sample["sharpness"]) / 100.0)
    return max(0.01, float(sample["det_score"]) * sharp_factor)


def observation_similarity(matching_samples, student_id, fallback):
    """Return the best nearby face similarity, or the track vote when none are nearby."""
    values = [float(sample["similarities"].get(student_id, -1.0))
              for sample in matching_samples]
    return max(values, default=float(fallback))


def aggregate_identity(sample_votes, threshold, margin, min_samples):
    """Quality-weighted mean similarity over multiple frames, with abstention."""
    if len(sample_votes) < min_samples:
        return {"student_id": None, "confidence": 0.0, "runner_up": 0.0,
                "margin": 0.0, "reason": "too_few_face_samples"}
    totals, weight_total = defaultdict(float), 0.0
    for similarities, weight in sample_votes:
        weight = max(0.0, float(weight))
        weight_total += weight
        for student_id, similarity in similarities.items():
            totals[student_id] += float(similarity) * weight
    if not weight_total or not totals:
        return {"student_id": None, "confidence": 0.0, "runner_up": 0.0,
                "margin": 0.0, "reason": "no_comparable_embeddings"}
    ranked = sorted(((score / weight_total, sid) for sid, score in totals.items()), reverse=True)
    best_score, best_id = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
    difference = best_score - runner_up
    if best_score < threshold:
        reason = "below_similarity_threshold"
    elif difference < margin:
        reason = "below_runner_up_margin"
    else:
        reason = "matched"
    return {"student_id": best_id if reason == "matched" else None,
            "best_candidate_id": best_id, "confidence": float(best_score),
            "runner_up": float(runner_up), "margin": float(difference), "reason": reason}


def resolve_overlapping_assignments(identity_rows):
    """For overlapping tracks assigned to one student, keep the higher score."""
    rows = [dict(row) for row in identity_rows]
    conflicts = []
    by_student = defaultdict(list)
    for row in rows:
        if row.get("student_id"):
            by_student[row["student_id"]].append(row)
    for student_id, candidates in by_student.items():
        candidates.sort(key=lambda r: r["identity_confidence"], reverse=True)
        kept = []
        for row in candidates:
            overlap_with = next((prior for prior in kept
                if min(row["last_seen_sec"], prior["last_seen_sec"])
                   >= max(row["first_seen_sec"], prior["first_seen_sec"])), None)
            if overlap_with is not None:
                conflicts.append({"student_id": student_id,
                    "kept_track_id": overlap_with["track_id"], "unknown_track_id": row["track_id"],
                    "kept_confidence": overlap_with["identity_confidence"],
                    "rejected_confidence": row["identity_confidence"],
                    "overlap_start_sec": max(row["first_seen_sec"], overlap_with["first_seen_sec"]),
                    "overlap_end_sec": min(row["last_seen_sec"], overlap_with["last_seen_sec"])})
                row["candidate_student_id"] = student_id
                row["student_id"] = None
                row["status"] = "unknown_conflict"
                row["reason"] = "overlapping_track_lower_confidence"
            else:
                kept.append(row)
    return rows, conflicts


def identity_stitches(identity_rows, max_gap):
    """Link adjacent, non-overlapping tracks for the same resolved student."""
    by_student = defaultdict(list)
    for row in identity_rows:
        if row.get("student_id"):
            by_student[row["student_id"]].append(row)
    groups, merges = [], []
    for student_id, rows in by_student.items():
        rows.sort(key=lambda r: (r["first_seen_sec"], r["track_id"]))
        current = None
        for row in rows:
            if current and 0 <= row["first_seen_sec"] - current["last_seen_sec"] <= max_gap:
                merges.append({"student_id": student_id, "from_track_id": current["track_ids"][-1],
                               "to_track_id": row["track_id"],
                               "gap_sec": row["first_seen_sec"] - current["last_seen_sec"]})
                current["track_ids"].append(row["track_id"])
                current["last_seen_sec"] = row["last_seen_sec"]
                current["identity_confidence"] = min(current["identity_confidence"], row["identity_confidence"])
                current["visibility_intervals"].extend(
                    row.get("visibility_intervals", [[row["first_seen_sec"], row["last_seen_sec"]]])
                )
            else:
                if current:
                    groups.append(current)
                current = {"student_id": student_id, "track_ids": [row["track_id"]],
                           "first_seen_sec": row["first_seen_sec"], "last_seen_sec": row["last_seen_sec"],
                           "identity_confidence": row["identity_confidence"],
                           "visibility_intervals": [[row["first_seen_sec"], row["last_seen_sec"]]]}
        if current:
            groups.append(current)
    return groups, merges


def choose_track_samples(track_rows, per_track, min_separation):
    """Pick large/confident boxes spread across each track's timeline."""
    chosen, last_time = [], []
    scored = []
    for row in track_rows.itertuples(index=False):
        width, height = max(0.0, row.x2 - row.x1), max(0.0, row.y2 - row.y1)
        score = width * height * (float(row.conf) if hasattr(row, "conf") else 1.0)
        scored.append((score, row))
    for _, row in sorted(scored, key=lambda pair: pair[0], reverse=True):
        if all(abs(float(row.t) - t) >= min_separation for t in last_time):
            chosen.append(row)
            last_time.append(float(row.t))
            if len(chosen) >= per_track:
                break
    return sorted(chosen, key=lambda row: row.t)


def visibility_intervals(times, step, gap_tolerance):
    """Convert sampled track times to intervals without claiming room entry/exit."""
    times = sorted(set(float(t) for t in times))
    if not times:
        return []
    intervals, start, previous = [], times[0], times[0]
    for current in times[1:]:
        if current - previous > gap_tolerance:
            intervals.append([start, previous + step])
            start = current
        previous = current
    intervals.append([start, previous + step])
    return intervals


def detect_track_faces(video_path, tracks, fps, detector, recognizer, gallery,
                       evidence_dir, config, time_min=None, time_max=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    samples_by_track, evidence_rows = defaultdict(list), []
    grouped_tracks = list(tracks.groupby("id"))
    total_tracks = len(grouped_tracks)
    for group_index, (track_id, group) in enumerate(grouped_tracks, start=1):
        eligible = group
        if time_min is not None:
            eligible = eligible[eligible.t.astype(float) >= time_min]
        if time_max is not None:
            eligible = eligible[eligible.t.astype(float) < time_max]
        selected = choose_track_samples(eligible, int(config["per_track_samples"]),
                                        float(config["min_sample_separation_sec"]))
        for candidate in selected:
            frame_number = max(0, int(round(float(candidate.t) * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            bw, bh = candidate.x2 - candidate.x1, candidate.y2 - candidate.y1
            x1 = max(0, int(candidate.x1 - 0.10 * bw))
            x2 = min(width, int(candidate.x2 + 0.10 * bw))
            y1 = max(0, int(candidate.y1 - 0.05 * bh))
            y2 = min(height, int(candidate.y1 + 0.48 * bh))
            head = frame[y1:y2, x1:x2]
            if head.size == 0:
                continue
            scale = 1.0
            if head.shape[1] < 200:
                scale = 200.0 / head.shape[1]
                head = cv2.resize(head, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            faces = detect_faces(detector, head)
            if not faces:
                continue
            face = max(faces, key=lambda item: float(item[2]) * float(item[3]))
            face_px = float(face[2]) / scale
            if face_px < float(config["min_face_px"]):
                continue
            fx, fy, fw, fh = (int(v) for v in face[:4])
            mx, my = int(0.2 * fw), int(0.2 * fh)
            crop = head[max(0, fy-my):min(head.shape[0], fy+fh+my),
                        max(0, fx-mx):min(head.shape[1], fx+fw+mx)]
            if crop.size == 0:
                continue
            sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if sharpness < float(config["min_sharpness"]):
                continue
            vector = embed(recognizer, head, face)
            similarities = {}
            for student_id, enrolled in gallery.items():
                similarities[student_id] = float(np.max(enrolled @ vector))
            filename = f"track_{int(track_id):05d}_t_{float(candidate.t):09.3f}.jpg"
            cv2.imwrite(str(evidence_dir / filename), crop)
            sample = {"t": float(candidate.t), "similarities": similarities,
                      "det_score": float(face[-1]), "sharpness": sharpness,
                      "face_px": face_px, "evidence_file": filename, "embedding": vector}
            samples_by_track[int(track_id)].append(sample)
            evidence_rows.append({"track_id": int(track_id), "t": float(candidate.t),
                "face_px": round(face_px, 1), "det_score": round(float(face[-1]), 3),
                "sharpness": round(sharpness, 1), "evidence_file": filename})
        if group_index == 1 or group_index % 5 == 0 or group_index == total_tracks:
            print(f"face sampling: tracks {group_index}/{total_tracks}", flush=True)
    cap.release()
    return samples_by_track, evidence_rows


def run_tracking(args, config, run_dir):
    command = [sys.executable, str(Path(__file__).with_name("track.py")),
        "--video", str(args.video), "--model", str(args.model or config["tracker_model"]),
        "--tracker", str(args.tracker or config["tracker_name"]),
        "--imgsz", str(args.imgsz or config["tracker_image_size"]),
        "--conf", str(args.track_conf or config["tracker_confidence"]),
        "--stride", str(args.stride or config["tracker_stride"]), "--out", str(run_dir)]
    subprocess.run(command, check=True)


def spatial_stitch(tracks_path, config):
    subprocess.run([sys.executable, str(Path(__file__).with_name("stitch.py")),
        "--tracks", str(tracks_path), "--max_gap", str(config["spatial_stitch_max_gap_sec"]),
        "--dist", str(config["spatial_stitch_distance_heights"])], check=True)
    return tracks_path.with_name("tracks_stitched.csv")


def build_demo_gallery(database, samples_by_track, duration_sec, enrollment_fraction=0.30):
    """Create a temporary, same-video identity gallery from early face samples."""
    cutoff = duration_sec * enrollment_fraction
    candidates = []
    for track_id, samples in samples_by_track.items():
        early = [sample for sample in samples if float(sample["t"]) < cutoff]
        if early:
            best = max(early, key=quality_weight)
            candidates.append((min(float(item["t"]) for item in early), int(track_id), best))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if not candidates:
        raise ValueError("No usable face samples were found in the first 30% of the video for demo identities")

    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    roster_rows = []
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY, name TEXT NOT NULL, roll_no TEXT NOT NULL,
                section TEXT NOT NULL, consent TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS enrolment_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL REFERENCES students(student_id),
                photo_name TEXT NOT NULL, embedding BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        connection.execute("DELETE FROM enrolment_embeddings")
        connection.execute("DELETE FROM students")
        for index, (_, track_id, sample) in enumerate(candidates, start=1):
            student_id = f"DEMO{index:03d}"
            name = f"Demo Student {index:02d}"
            roll_no = f"DEMO-{index:02d}"
            connection.execute(
                "INSERT INTO students(student_id,name,roll_no,section,consent) VALUES(?,?,?,?,?)",
                (student_id, name, roll_no, "DEMO", "yes"))
            vector = np.asarray(sample["embedding"], dtype=np.float32).reshape(-1)
            connection.execute(
                "INSERT INTO enrolment_embeddings(student_id,photo_name,embedding) VALUES(?,?,?)",
                (student_id, f"track_{track_id}_early_sample", vector.tobytes()))
            roster_rows.append({"student_id": student_id, "name": name, "roll_no": roll_no,
                                "section": "DEMO", "source_track_id": track_id,
                                "demo_label": "DEMO: same-video enrollment; not an accuracy evaluation"})
        connection.commit()
    pd.DataFrame(roster_rows).to_csv(database.with_name("demo_students.csv"), index=False)
    return cutoff, len(roster_rows)


def export_results(run_dir, tracks, meta, samples_by_track, evidence_rows, gallery, config, demo=False):
    identities = []
    for track_id, group in tracks.groupby("id"):
        tmin, tmax = float(group.t.min()), float(group.t.max())
        intervals = visibility_intervals(group.t, float(meta.get("step", 1)),
                                         float(config["visibility_gap_tolerance_sec"]))
        samples = samples_by_track.get(int(track_id), [])
        votes = [(sample["similarities"], quality_weight(sample)) for sample in samples]
        result = aggregate_identity(votes, float(config["identity_threshold"]),
                                    float(config["identity_margin"]), int(config["min_face_samples"]))
        assigned = result.get("student_id")
        best_sample = max(samples, key=quality_weight) if samples else None
        identity = {"track_id": int(track_id), "orig_track_ids": sorted(set(int(x) for x in group.get("orig_id", group.id))),
            "student_id": assigned, "candidate_student_id": result.get("best_candidate_id"),
            "status": "identified" if assigned else "unknown", "reason": result["reason"],
            "identity_confidence": round(float(result["confidence"]), 6),
            "mean_person_detection_confidence": round(float(pd.to_numeric(
                group.get("conf", pd.Series(dtype=float)), errors="coerce").mean()), 4)
                if "conf" in group and pd.to_numeric(group["conf"], errors="coerce").notna().any() else 0.0,
            "runner_up_similarity": round(float(result["runner_up"]), 6),
            "identity_margin": round(float(result["margin"]), 6), "face_sample_count": len(samples),
            "first_seen_sec": tmin, "last_seen_sec": tmax,
            "demo_mode": bool(demo),
            "demo_label": "DEMO: same-video enrollment; not an accuracy evaluation" if demo else "",
            "visibility_intervals": intervals,
            "best_face_file": best_sample["evidence_file"] if best_sample else None,
            "best_face_time_sec": best_sample["t"] if best_sample else None}
        identities.append(identity)
    identities, conflicts = resolve_overlapping_assignments(identities)
    identity_groups, identity_merges = identity_stitches(
        identities, float(config["identity_stitch_max_gap_sec"]))
    # Keep the highest quality unknown face as a named standalone evidence image.
    unknown_dir = run_dir / "unknown_faces"
    unknown_dir.mkdir(exist_ok=True)
    for identity in identities:
        if identity["student_id"] is None and identity["best_face_file"]:
            source = run_dir / "face_evidence" / identity["best_face_file"]
            target = unknown_dir / f"unknown_track_{identity['track_id']:05d}.jpg"
            shutil.copyfile(source, target)
            identity["best_face_file"] = str(target.relative_to(run_dir))
    (run_dir / "identities.json").write_text(json.dumps(identities, indent=2), encoding="utf-8")
    (run_dir / "identity_stitched_tracks.json").write_text(json.dumps(identity_groups, indent=2), encoding="utf-8")
    (run_dir / "session_results.json").write_text(json.dumps({
        "enrolment_gallery": ("DEMO: same-video enrolment from first 30%; not an accuracy evaluation"
                              if demo else "SQLite embeddings for consent=yes students"),
        "demo_mode": bool(demo),
        "demo_label": "DEMO: same-video enrollment; not an accuracy evaluation" if demo else None,
        "time_basis": "video-relative seconds",
        "duration_sec": float(meta.get("duration", 0)), "tracks": identities,
        "identity_stitched_tracks": identity_groups, "conflicts": conflicts}, indent=2), encoding="utf-8")
    pd.DataFrame(identities).to_csv(run_dir / "identities.csv", index=False)
    pd.DataFrame([row for row in identities if row["student_id"] is None]).to_csv(run_dir / "unknown_tracks.csv", index=False)
    pd.DataFrame(conflicts, columns=["student_id", "kept_track_id", "unknown_track_id", "kept_confidence",
        "rejected_confidence", "overlap_start_sec", "overlap_end_sec"]).to_csv(run_dir / "conflicts.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(run_dir / "face_samples.csv", index=False)
    pd.DataFrame(identity_merges, columns=["student_id", "from_track_id", "to_track_id", "gap_sec"]).to_csv(
        run_dir / "identity_merges.csv", index=False)
    # Stable input shape for the later presence report: one row per identified observation.
    presence_rows = []
    by_identity = {row["track_id"]: row for row in identities}
    for track_id, group in tracks.groupby("id"):
        identity = by_identity[int(track_id)]
        if not identity["student_id"]:
            continue
        for t in sorted(group.t.astype(float).unique()):
            matching = [s for s in samples_by_track.get(int(track_id), []) if abs(s["t"] - t) <= float(meta.get("step", 1))]
            sim = observation_similarity(matching, identity["student_id"],
                                         identity["identity_confidence"])
            presence_rows.append({"student_id": identity["student_id"], "track_id": int(track_id),
                "t": t, "similarity": sim, "face_detected": int(bool(matching)),
                "identity_confidence": identity["identity_confidence"]})
    pd.DataFrame(presence_rows, columns=["student_id", "track_id", "t", "similarity", "face_detected",
        "identity_confidence"]).to_csv(run_dir / "presence_input.csv", index=False)
    print(f"M2 complete: {len(identities)} tracks, {sum(bool(r['student_id']) for r in identities)} identified, "
          f"{sum(r['student_id'] is None for r in identities)} unknown, {len(conflicts)} identity conflicts")
    print(f"Results: {run_dir}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    external_models = project_root.parent / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=project_root / "data" / "attendance.db")
    parser.add_argument("--out", type=Path, default=project_root / "output" / "session_m2")
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--det", type=Path, default=external_models / "yunet.onnx")
    parser.add_argument("--rec", type=Path, default=external_models / "sface.onnx")
    parser.add_argument("--model", default=None, help="YOLO weights; defaults to config.yaml")
    parser.add_argument("--tracker", default=None, help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--track-conf", type=float, default=None)
    parser.add_argument("--identity-threshold", type=float, default=None)
    parser.add_argument("--identity-margin", type=float, default=None)
    parser.add_argument("--reuse-tracks", action="store_true", help="use tracks.csv already in --out")
    parser.add_argument("--demo", action="store_true",
                        help="create temporary same-video demo identities; not an accuracy evaluation")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    if not args.demo and not args.db.is_file():
        parser.error(f"enrolment database not found: {args.db}")
    config = load_config(args.config)
    if args.identity_threshold is not None:
        config["identity_threshold"] = args.identity_threshold
    if args.identity_margin is not None:
        config["identity_margin"] = args.identity_margin
    run_dir = args.out.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = run_dir / "tracks.csv"
    if not args.reuse_tracks:
        run_tracking(args, config, run_dir)
    if not tracks_path.is_file():
        parser.error(f"tracks file not found: {tracks_path}")
    stitched_path = spatial_stitch(tracks_path, config)
    tracks = pd.read_csv(stitched_path)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    detector, recognizer = make_models(str(args.det), str(args.rec),
                                       float(config["face_detection_score"]))
    if args.demo:
        cutoff = float(meta.get("duration", 0)) * 0.30
        print("DEMO enrollment pass: sampling the first 30% of the video", flush=True)
        early_samples, _ = detect_track_faces(args.video, tracks, float(meta.get("fps", 25)),
            detector, recognizer, {}, run_dir / "face_evidence", config, time_max=cutoff)
        demo_database = run_dir / "demo_gallery.db"
        cutoff, demo_count = build_demo_gallery(demo_database, early_samples, float(meta.get("duration", 0)))
        gallery = load_gallery(demo_database)
        print("DEMO recognition pass: sampling the remaining video", flush=True)
        samples, evidence = detect_track_faces(args.video, tracks, float(meta.get("fps", 25)),
            detector, recognizer, gallery, run_dir / "face_evidence", config, time_min=cutoff)
        print(f"DEMO: {demo_count} temporary identities enrolled from the first 30%; "
              "same-video enrolment, not an accuracy evaluation")
    else:
        gallery = load_gallery(args.db)
        print("Face sampling and recognition", flush=True)
        samples, evidence = detect_track_faces(args.video, tracks, float(meta.get("fps", 25)),
            detector, recognizer, gallery, run_dir / "face_evidence", config)
    export_results(run_dir, tracks, meta, samples, evidence, gallery, config, demo=args.demo)


if __name__ == "__main__":
    main()
