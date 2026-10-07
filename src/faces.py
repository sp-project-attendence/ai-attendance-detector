"""WP2 optional face layer, using OpenCV's built-in YuNet (detector) and SFace (recognizer).

Three commands:
  extract : for each track, find its best faces and compute a face embedding
  enrol   : build a gallery (name -> embedding) from labelled photos  (photos/<Name>/*.jpg)
  match   : name each track using the gallery, and suggest tracks that look like the same person

Model files: download YuNet and SFace .onnx from the OpenCV Zoo and save them as
  ../models/yunet.onnx   and   ../models/sface.onnx      (relative to the repo folder)
"""
import argparse, glob, os
import cv2
import numpy as np
import pandas as pd

DEF_DET = os.path.join("..", "models", "yunet.onnx")
DEF_REC = os.path.join("..", "models", "sface.onnx")


def make_models(det_path, rec_path, score=0.6):
    for p in (det_path, rec_path):
        if not os.path.exists(p):
            raise SystemExit(f"Model file not found: {p}")
        if os.path.getsize(p) < 10000:
            raise SystemExit(f"{p} is only {os.path.getsize(p)} bytes: probably a download pointer, "
                             "not the real model. Download it again with the download button.")
    det = cv2.FaceDetectorYN.create(det_path, "", (320, 320), score, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(rec_path, "")
    return det, rec


def detect_faces(det, img):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    return [] if faces is None else list(faces)


def embed(rec, img, face):
    aligned = rec.alignCrop(img, face)
    f = rec.feature(aligned).flatten().astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-9)


def cmd_extract(a):
    det, rec = make_models(a.det, a.rec)
    df = pd.read_csv(a.tracks)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    df["score"] = (df.y2 - df.y1) * (df["conf"] if "conf" in df else 1.0)

    chosen = []   # biggest/most confident boxes per track, spread over time
    for tid, d in df.groupby("id"):
        picked = []
        for r in d.sort_values("score", ascending=False).itertuples():
            if all(abs(r.t - pt) >= a.min_sep for pt in picked):
                picked.append(r.t)
                chosen.append((int(round(r.t * fps)), int(tid), r.x1, r.y1, r.x2, r.y2, r.t))
            if len(picked) >= a.per_track:
                break
    by_frame = {}
    for c in chosen:
        by_frame.setdefault(c[0], []).append(c)

    os.makedirs(a.out, exist_ok=True)
    rows, embs, counts = [], {}, {}
    idx, last = 0, (max(by_frame) if by_frame else -1)
    while idx <= last:
        if idx in by_frame:
            ok, frame = cap.read()
        else:
            ok, frame = cap.grab(), None
        if not ok:
            break
        if frame is not None:
            H, W = frame.shape[:2]
            for (_, tid, x1, y1, x2, y2, t) in by_frame[idx]:
                bw, bh = x2 - x1, y2 - y1
                crop = frame[int(max(0, y1 - 0.05 * bh)):int(min(H, y1 + 0.45 * bh)),
                             int(max(0, x1 - 0.1 * bw)):int(min(W, x2 + 0.1 * bw))]   # head area
                if crop.size == 0:
                    continue
                scale = 1.0
                if crop.shape[1] < 200:      # tiny head crops: enlarge so the detector can see them
                    scale = 200.0 / crop.shape[1]
                    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                faces = detect_faces(det, crop)
                if not faces:
                    continue
                f = max(faces, key=lambda r: r[-1])
                face_px = float(f[2]) / scale
                if face_px < a.min_face:
                    continue
                x, y, w_, h_ = [int(v) for v in f[:4]]
                mx, my = int(0.2 * w_), int(0.2 * h_)
                face_img = crop[max(0, y - my):y + h_ + my, max(0, x - mx):x + w_ + mx]
                if face_img.size == 0:
                    continue
                sharp = cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                k = counts.get(tid, 0)
                counts[tid] = k + 1
                fn = f"id_{tid:03d}_{k}.jpg"
                cv2.imwrite(os.path.join(a.out, fn), face_img)
                embs.setdefault(tid, []).append((embed(rec, crop, f), float(f[-1])))
                rows.append(dict(id=tid, t=round(t, 2), face_px=round(face_px, 1),
                                 det_score=round(float(f[-1]), 3), sharpness=round(float(sharp), 1), file=fn))
        idx += 1

    ids = sorted(embs)
    means = []
    for tid in ids:
        arr = np.array([e for e, _ in embs[tid]])
        w = np.array([s for _, s in embs[tid]])
        m = (arr * w[:, None]).sum(0)
        means.append(m / (np.linalg.norm(m) + 1e-9))
    np.savez(os.path.join(a.out, "track_embeddings.npz"), ids=np.array(ids), embs=np.array(means),
             counts=np.array([len(embs[i]) for i in ids]))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "faces.csv"), index=False)
    all_ids = sorted(df["id"].unique())
    missing = [int(i) for i in all_ids if i not in embs]
    print(f"tracks: {len(all_ids)} | with a usable face: {len(ids)} | faces saved: {len(rows)}")
    print(f"tracks with no usable face: {missing}")


def cmd_enrol(a):
    det, rec = make_models(a.det, a.rec)
    names, embs = [], []
    for folder in sorted(glob.glob(os.path.join(a.photos, "*"))):
        if not os.path.isdir(folder):
            continue
        vs = []
        for p in glob.glob(os.path.join(folder, "*")):
            img = cv2.imread(p)
            if img is None:
                continue
            s = 1280.0 / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            faces = detect_faces(det, img)
            if not faces:
                print("  no face found:", p)
                continue
            vs.append(embed(rec, img, max(faces, key=lambda r: r[2] * r[3])))
        if vs:
            m = np.mean(vs, axis=0)
            names.append(os.path.basename(folder))
            embs.append(m / (np.linalg.norm(m) + 1e-9))
            print(f"{os.path.basename(folder)}: {len(vs)} photos used")
    np.savez(a.out, names=np.array(names), embs=np.array(embs))
    print("saved gallery:", a.out)


def cmd_match(a):
    te = np.load(os.path.join(a.faces, "track_embeddings.npz"))
    ids, E, counts = te["ids"], te["embs"], te["counts"]
    tr = pd.read_csv(a.tracks) if a.tracks else None
    rows = []
    if a.gallery:
        g = np.load(a.gallery)
        names, G = g["names"], g["embs"]
        sims = E @ G.T
        for i, tid in enumerate(ids):
            order = np.argsort(-sims[i])
            s1 = float(sims[i][order[0]])
            s2 = float(sims[i][order[1]]) if len(order) > 1 else -1.0
            named = s1 >= a.threshold and (s1 - s2) >= a.margin
            rows.append(dict(id=int(tid), name=str(names[order[0]]) if named else "unknown",
                             best_guess=str(names[order[0]]), similarity=round(s1, 3),
                             runner_up=round(s2, 3), n_faces=int(counts[i]),
                             status="named" if named else "uncertain"))
    if tr is not None:
        for tid in sorted(set(tr["id"].unique()) - set(int(i) for i in ids)):
            rows.append(dict(id=int(tid), name="unknown", status="no_face"))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.faces, "identities.csv"), index=False)
        print("wrote identities.csv")
    if tr is not None:     # tracks that look like the same person and never overlap in time
        span = tr.groupby("id")["t"].agg(["min", "max"])
        S = E @ E.T
        sug = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ia, ib = int(ids[i]), int(ids[j])
                if ia not in span.index or ib not in span.index:
                    continue
                overlap = min(span.loc[ia, "max"], span.loc[ib, "max"]) - max(span.loc[ia, "min"], span.loc[ib, "min"])
                if overlap > 0.5:
                    continue      # both on screen at once, so they are different people
                if S[i, j] >= a.merge_thr:
                    sug.append(dict(id_a=ia, id_b=ib, face_similarity=round(float(S[i, j]), 3)))
        pd.DataFrame(sug).to_csv(os.path.join(a.faces, "merge_suggestions.csv"), index=False)
        print(f"wrote merge_suggestions.csv ({len(sug)} pairs)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "enrol"):
        p = sub.add_parser(name)
        p.add_argument("--det", default=DEF_DET)
        p.add_argument("--rec", default=DEF_REC)
        if name == "extract":
            p.add_argument("--video", required=True)
            p.add_argument("--tracks", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--per_track", type=int, default=8)
            p.add_argument("--min_sep", type=float, default=0.5)
            p.add_argument("--min_face", type=float, default=20, help="min face width in original pixels")
        else:
            p.add_argument("--photos", required=True)
            p.add_argument("--out", required=True)
    p = sub.add_parser("match")
    p.add_argument("--faces", required=True, help="folder written by 'extract'")
    p.add_argument("--tracks", default=None)
    p.add_argument("--gallery", default=None)
    p.add_argument("--threshold", type=float, default=0.363, help="cosine similarity needed to name a track (tune it!)")
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--merge_thr", type=float, default=0.5)
    a = ap.parse_args()
    {"extract": cmd_extract, "enrol": cmd_enrol, "match": cmd_match}[a.cmd](a)


if __name__ == "__main__":
    main()
