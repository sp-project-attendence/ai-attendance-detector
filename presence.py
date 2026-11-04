"""Generate per-student presence reports from M2's presence_input.csv.

Confidence formula (0–100): 50% * mean recognition similarity + 30% * the
percentage of visible sampled time with a detectable face + 20% * sample
adequacy. Sample adequacy is min(face_samples / 5, 1). Each input component
is clamped to 0–1. The formula describes evidence quality; it is not an
attendance decision.
"""
import argparse
import json
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd


DEFAULTS = {
    "gap_tol_sec": 15.0,
    "late_first_seen_fraction": 0.10,
    "early_last_seen_fraction": 0.10,
    "multiple_gaps_min_count": 2,
    "long_gap_sec": 60.0,
    "low_visibility_percent": 40.0,
    "medium_confidence_score": 55.0,
    "high_confidence_score": 75.0,
    "confidence_sample_target": 5,
}


def load_config(path):
    """Read this project's simple flat YAML key:value settings without a new dependency."""
    values = dict(DEFAULTS)
    if not path or not Path(path).is_file():
        return values
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw = (part.strip() for part in line.split(":", 1))
        try:
            values[key] = float(raw) if any(c in raw for c in ".eE") else int(raw)
        except ValueError:
            continue
    return values


def read_roster(database=None, students_csv=None):
    """Read all consenting M1 students, or a synthetic CSV roster for a smoke test."""
    if students_csv:
        students = pd.read_csv(students_csv, dtype={"student_id": str})
        if "student_id" not in students or "name" not in students:
            raise ValueError("Roster CSV must include student_id and name columns")
        if "consent" in students:
            students = students[students.consent.fillna("").str.strip().str.lower() == "yes"]
        return students.drop_duplicates("student_id").reset_index(drop=True)
    if not database:
        raise ValueError("Provide --db for M1 students or --students-csv for synthetic input")
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query("""
            SELECT student_id, name, roll_no, section
            FROM students WHERE lower(trim(consent))='yes'
            ORDER BY student_id
        """, connection)


def parse_clock(value):
    """Parse HH:MM[:SS] or an ISO datetime into a datetime for clock arithmetic."""
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = time.fromisoformat(text)
        return datetime.combine(date.today(), parsed)


def session_timing(session_json=None, duration_override=None, meta_path=None):
    """Return session duration, video clock anchor, and whether schedule flags apply."""
    session = json.loads(Path(session_json).read_text(encoding="utf-8")) if session_json else {}
    scheduled = bool(session.get("scheduled_start") and session.get("scheduled_end"))
    if scheduled:
        start = parse_clock(session["scheduled_start"])
        end = parse_clock(session["scheduled_end"])
        if end <= start:
            end += timedelta(days=1)
        duration = (end - start).total_seconds()
        video_start = parse_clock(session.get("video_start", session["scheduled_start"]))
        if video_start.date() == date.today():
            video_start = video_start.replace(year=start.year, month=start.month, day=start.day)
        if video_start < start - timedelta(hours=12):
            video_start += timedelta(days=1)
        return duration, video_start, True
    if duration_override is not None:
        return float(duration_override), None, False
    if meta_path and Path(meta_path).is_file():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return float(meta["duration"]), None, False
    return 600.0, None, False


def read_optional_csv(path):
    """Treat an empty optional M2 sidecar as an empty table."""
    if not Path(path).is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def format_time(seconds, clock_anchor=None):
    seconds = max(0.0, float(seconds))
    if clock_anchor is None:
        total = int(seconds)
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    return (clock_anchor + timedelta(seconds=seconds)).strftime("%H:%M:%S")


def make_intervals(times, step_sec, duration_sec, gap_tol_sec):
    """Build intervals and merge any unobserved gap strictly shorter than GAP_TOL."""
    points = sorted(set(float(t) for t in times if pd.notna(t) and 0 <= float(t) <= duration_sec))
    if not points:
        return []
    intervals = []
    start = previous = points[0]
    for current in points[1:]:
        previous_end = min(duration_sec, previous + step_sec)
        gap = current - previous_end
        if gap >= gap_tol_sec:
            intervals.append([round(start, 3), round(previous_end, 3)])
            start = current
        previous = current
    intervals.append([round(start, 3), round(min(duration_sec, previous + step_sec), 3)])
    return intervals


def _extract_sample_metrics(student_id, observations, face_samples, identities, step_sec):
    """Use M2 sidecars to count extracted crops and estimate per-crop similarity."""
    if face_samples is None or identities is None or not {"track_id", "student_id"}.issubset(identities.columns):
        selected = observations[observations.face_detected > 0].drop_duplicates("t")
        sims = pd.to_numeric(selected.get("similarity", pd.Series(dtype=float)), errors="coerce").dropna()
        return int(len(selected)), sims.tolist()
    track_ids = identities.loc[identities.student_id == student_id, "track_id"].astype(str).tolist()
    if not track_ids or face_samples.empty:
        return 0, []
    crops = face_samples[face_samples.track_id.astype(str).isin(track_ids)]
    similarities = []
    for crop in crops.itertuples(index=False):
        same_track = observations[observations.track_id.astype(str) == str(crop.track_id)]
        if same_track.empty:
            continue
        distances = (pd.to_numeric(same_track.t, errors="coerce") - float(crop.t)).abs()
        nearest_index = distances.idxmin()
        if float(distances.loc[nearest_index]) <= max(step_sec, 0.001):
            value = pd.to_numeric(pd.Series([same_track.loc[nearest_index, "similarity"]]), errors="coerce").iloc[0]
            if pd.notna(value):
                similarities.append(float(value))
    return int(len(crops)), similarities


def build_presence(students, observations, duration_sec, step_sec, config=None,
                   scheduled=False, clock_anchor=None, face_samples=None, identities=None):
    """Return report rows and a per-student timeline mapping."""
    cfg = {**DEFAULTS, **(config or {})}
    required = {"student_id", "t"}
    if not required.issubset(observations.columns):
        raise ValueError("presence_input.csv must include student_id and t columns")
    if "face_detected" not in observations:
        observations["face_detected"] = 0
    if "similarity" not in observations:
        observations["similarity"] = float("nan")
    observations = observations.copy()
    observations["t"] = pd.to_numeric(observations.t, errors="coerce")
    observations["face_detected"] = pd.to_numeric(observations.face_detected, errors="coerce").fillna(0)
    if "track_id" not in observations:
        observations["track_id"] = ""

    rows, timelines = [], {}
    for student in students.itertuples(index=False):
        sid = str(student.student_id)
        seen = observations[observations.student_id.astype(str) == sid].dropna(subset=["t"])
        # A single person contributes at most one visibility sample per timestamp.
        seen = (seen.sort_values(["t", "face_detected"])
                .groupby("t", as_index=False).agg({
                    "face_detected": "max", "similarity": "max", "track_id": "first"}))
        intervals = make_intervals(seen.t.tolist(), step_sec, duration_sec, float(cfg["gap_tol_sec"]))
        total_visible = sum(end - start for start, end in intervals)
        visible_pct = min(100.0, 100 * total_visible / duration_sec) if duration_sec > 0 else 0.0
        first_sec = float(seen.t.min()) if not seen.empty else None
        last_sec = float(seen.t.max()) if not seen.empty else None
        gaps = [max(0.0, intervals[i + 1][0] - intervals[i][1]) for i in range(len(intervals) - 1)]
        gap_count = len(gaps)
        longest_gap = max(gaps, default=0.0)

        samples_count, sample_similarities = _extract_sample_metrics(
            sid, seen, face_samples, identities, step_sec)
        if sample_similarities:
            mean_similarity = sum(sample_similarities) / len(sample_similarities)
        else:
            detected = seen[seen.face_detected > 0].drop_duplicates("t")
            similarity_values = pd.to_numeric(detected.similarity, errors="coerce").dropna()
            mean_similarity = float(similarity_values.mean()) if len(similarity_values) else 0.0
            if samples_count == 0:
                samples_count = len(detected)
        face_observation_count = int(seen.face_detected.gt(0).sum())
        estimated_face_seconds = min(total_visible, face_observation_count * step_sec)
        face_visible_pct = 100 * estimated_face_seconds / total_visible if total_visible > 0 else 0.0
        mean_person_detection = 0.0
        if (identities is not None and
                {"track_id", "student_id", "mean_person_detection_confidence"}.issubset(identities.columns)):
            student_tracks = identities[identities.student_id.astype(str) == sid]
            values = pd.to_numeric(student_tracks.mean_person_detection_confidence, errors="coerce").dropna()
            if len(values):
                mean_person_detection = float(values.mean())

        sim_component = min(1.0, max(0.0, mean_similarity))
        face_component = min(1.0, max(0.0, face_visible_pct / 100))
        sample_target = max(1, int(cfg["confidence_sample_target"]))
        sample_component = min(1.0, samples_count / sample_target)
        score = round(100 * (0.50 * sim_component + 0.30 * face_component + 0.20 * sample_component), 1)
        if score >= float(cfg["high_confidence_score"]):
            confidence = "High"
        elif score >= float(cfg["medium_confidence_score"]):
            confidence = "Medium"
        else:
            confidence = "Low"
        status = "Not seen" if not intervals else "Seen (low confidence)" if confidence == "Low" else "Seen"

        flags = []
        if scheduled and first_sec is not None and first_sec > duration_sec * float(cfg["late_first_seen_fraction"]):
            flags.append("Late first seen")
        # The video itself is the observation window even when clock schedule data is absent.
        # This indicates early loss of visibility, not proof that the student left.
        early_fraction = float(cfg["early_last_seen_fraction"])
        started_near_video_start = first_sec is not None and first_sec <= duration_sec * early_fraction
        ended_well_before_video_end = last_sec is not None and last_sec < duration_sec * (1 - early_fraction)
        if started_near_video_start and ended_well_before_video_end:
            flags.append("Early last seen")
        if gap_count >= int(cfg["multiple_gaps_min_count"]):
            flags.append("Multiple gaps")
        if longest_gap >= float(cfg["long_gap_sec"]):
            flags.append("Long gap")
        if visible_pct < float(cfg["low_visibility_percent"]):
            flags.append("Low visibility")
        if confidence == "Low":
            flags.append("Low confidence")

        first_seen = format_time(first_sec, clock_anchor) if first_sec is not None else None
        last_seen = format_time(last_sec, clock_anchor) if last_sec is not None else None
        student_name = getattr(student, "name", sid)
        row = {
            "student_id": sid, "name": student_name,
            "first_seen": first_seen, "last_seen": last_seen,
            "first_seen_sec": first_sec, "last_seen_sec": last_sec,
            "total_visible_seconds": round(total_visible, 2), "visible_percent": round(visible_pct, 2),
            "visibility_intervals": json.dumps(intervals), "gap_count": gap_count,
            "longest_gap_seconds": round(longest_gap, 2),
            "mean_recognition_similarity": round(mean_similarity, 4),
            "face_detected_visible_percent": round(face_visible_pct, 2),
            "mean_person_detection_confidence": round(mean_person_detection, 4),
            "face_samples": int(samples_count), "confidence_score": score,
            "confidence": confidence, "status": status, "flags": ", ".join(flags) if flags else "—",
        }
        rows.append(row)
        timelines[sid] = {
            "student_id": sid, "name": student_name,
            "first_seen": first_seen, "last_seen": last_seen,
            "first_seen_sec": first_sec, "last_seen_sec": last_sec,
            "visibility_intervals": [{"start_sec": start, "end_sec": end,
                                      "duration_sec": round(end - start, 2)} for start, end in intervals],
            "gap_count": gap_count, "longest_gap_seconds": round(longest_gap, 2),
            "confidence_components": {
                "mean_recognition_similarity": round(mean_similarity, 4),
                "face_detected_visible_percent": round(face_visible_pct, 2),
                "mean_person_detection_confidence": round(mean_person_detection, 4),
                "face_samples": int(samples_count),
            },
            "confidence_score": score, "confidence": confidence,
            "status": status, "flags": flags,
        }
    return pd.DataFrame(rows), timelines


def write_summary(report, destination, duration_sec, time_basis, demo=False):
    lines = [
        "# Session presence summary", "",
        *( ["**DEMO: same-video enrollment; not an accuracy evaluation.**", ""] if demo else [] ),
        f"Session duration used for percentages: {duration_sec:.1f} seconds ({time_basis}).", "",
        "**Visibility is evidence from this video only. Not visible does NOT mean the student left the room. "
        "A person may be occluded or not detectable. An early last-seen flag means visibility stopped before "
        "the video ended; it does not confirm departure. This report does not decide attendance.**", "",
        "## Confidence formula", "",
        "Score = 100 × (0.50 × mean recognition similarity + 0.30 × detectable-face visible-time fraction "
        "+ 0.20 × min(face samples / 5, 1)). Inputs are clamped to 0–1. "
        "High ≥ 75, Medium ≥ 55, Low < 55. These are evidence-quality labels, not identity guarantees.", "",
        "## Students", "",
    ]
    for row in report.to_dict(orient="records"):
        lines.extend([
            f"### {row['name']} ({row['student_id']})", "",
            f"- Status: {row['status']}",
            f"- First seen: {row['first_seen'] or 'Not seen'}; last seen: {row['last_seen'] or 'Not seen'}",
            f"- Visible: {row['total_visible_seconds']:.1f} sec ({row['visible_percent']:.1f}% of session)",
            f"- Gaps: {row['gap_count']}; longest gap: {row['longest_gap_seconds']:.1f} sec",
            f"- Confidence: {row['confidence']} ({row['confidence_score']:.1f}/100); flags: {row['flags']}", "",
        ])
    Path(destination).write_text("\n".join(lines), encoding="utf-8")


def generate_report(input_dir, output_dir, database=None, students_csv=None,
                    duration_sec=None, session_json=None, config_path=None, demo=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    input_path = input_dir if input_dir.is_file() else None
    folder = input_path.parent if input_path else input_dir
    presence_path = input_path or (folder / "presence_input.csv")
    if not presence_path.is_file():
        # Preserve Step A's earlier synthetic input format.
        presence_path = folder / "observations.csv"
    observations = pd.read_csv(presence_path)
    students = read_roster(database, students_csv or (folder / "students.csv" if (folder / "students.csv").is_file() else None))
    if duration_sec is None:
        meta_path = folder / "meta.json"
    else:
        meta_path = None
    duration_sec, clock_anchor, scheduled = session_timing(session_json, duration_sec, meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path and Path(meta_path).is_file() else {}
    step_sec = float(meta.get("step", 5.0))
    config = load_config(config_path)
    face_samples_path, identities_path = folder / "face_samples.csv", folder / "identities.csv"
    face_samples = read_optional_csv(face_samples_path)
    identities = read_optional_csv(identities_path)
    report, timelines = build_presence(students, observations, duration_sec, step_sec, config,
        scheduled, clock_anchor, face_samples, identities)
    if demo:
        label = "DEMO: same-video enrollment; not an accuracy evaluation"
        report["demo_mode"] = True
        report["demo_label"] = label
        for timeline in timelines.values():
            timeline["demo_mode"] = True
            timeline["demo_label"] = label
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "report.csv", index=False)
    (output_dir / "timelines.json").write_text(json.dumps(timelines, indent=2), encoding="utf-8")
    time_basis = "scheduled class duration" if scheduled else "video-relative duration; no schedule supplied"
    if demo:
        time_basis = "relative times, demo; " + time_basis
    write_summary(report, output_dir / "session_summary.md", duration_sec, time_basis, demo=demo)
    print(f"Presence report: {output_dir / 'report.csv'} ({len(report)} roster students)")
    return report, timelines


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "output" / "session_m2",
                        help="M2 output folder or presence_input.csv; defaults to Step A simulated folder if M2 folder is absent")
    parser.add_argument("--out", type=Path, default=None, help="report output folder (defaults to input folder)")
    parser.add_argument("--db", type=Path, default=project_root / "data" / "attendance.db")
    parser.add_argument("--students-csv", type=Path, default=None, help="synthetic/test roster CSV instead of SQLite")
    parser.add_argument("--duration-sec", type=float, default=None, help="scheduled session duration when no session JSON is supplied")
    parser.add_argument("--session-json", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    parser.add_argument("--demo", action="store_true", help="label outputs as same-video demo, not accuracy evaluation")
    args = parser.parse_args()
    input_dir = args.input
    if not input_dir.exists() and args.input == project_root / "output" / "session_m2":
        input_dir = project_root / "output" / "simulated"
    out_dir = args.out or (input_dir.parent if input_dir.is_file() else input_dir)
    generate_report(input_dir, out_dir, args.db if args.db.is_file() else None,
        args.students_csv, args.duration_sec, args.session_json, args.config, demo=args.demo)


if __name__ == "__main__":
    main()
