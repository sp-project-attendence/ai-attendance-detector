"""WP2: video -> YOLO detection -> tracker IDs -> tracks.csv, meta.json, annotated.mp4, one best crop per ID.

Example:
  python src/track.py --video E:/zsp_project/attendance-data/kids.mp4 --model yolo11n.pt \
      --imgsz 640 --stride 5 --out E:/zsp_project/attendance-data/run1_byte
"""
import argparse, json, os, subprocess, time
import cv2
import pandas as pd
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolo11n.pt")        # n = fastest, s = better, m = best/slowest
    ap.add_argument("--tracker", default="bytetrack.yaml")  # or botsort.yaml
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=5)        # process every Nth frame
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    crop_dir = os.path.join(a.out, "crops")
    os.makedirs(crop_dir, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    raw_path = os.path.join(a.out, "annotated_raw.mp4")
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (W, H))
    model = YOLO(a.model)

    rows = []
    best = {}  # track id -> (score, crop image)
    t0 = time.time()
    results = model.track(source=a.video, stream=True, persist=True, tracker=a.tracker,
                          classes=[0], conf=a.conf, imgsz=a.imgsz,
                          vid_stride=a.stride, verbose=False)
    for i, r in enumerate(results):
        t = (i * a.stride) / fps
        if r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            boxes = r.boxes.xyxy.tolist()
            confs = r.boxes.conf.tolist()
            for tid, (x1, y1, x2, y2), cf in zip(ids, boxes, confs):
                rows.append(dict(t=round(t, 3), id=tid, x1=round(x1, 1), y1=round(y1, 1),
                                 x2=round(x2, 1), y2=round(y2, 1), conf=round(cf, 3)))
                # best crop = large, confident, not cut off by the frame edge
                score = cf * (x2 - x1) * (y2 - y1)
                if x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2:
                    score *= 0.5
                if tid not in best or score > best[tid][0]:
                    cx1, cy1 = max(int(x1), 0), max(int(y1), 0)
                    cx2, cy2 = min(int(x2), W), min(int(y2), H)
                    if cx2 > cx1 and cy2 > cy1:
                        best[tid] = (score, r.orig_img[cy1:cy2, cx1:cx2].copy())
        writer.write(r.plot())
        if i % 20 == 0:
            print(f"processed {i * a.stride}/{n_frames} frames", end="\r")
    writer.release()
    elapsed = time.time() - t0

    for tid, (_, img) in best.items():
        cv2.imwrite(os.path.join(crop_dir, f"id_{tid:03d}.jpg"), img)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "tracks.csv"), index=False)
    meta = dict(fps=fps, duration=n_frames / fps, step=a.stride / fps,
                model=a.model, tracker=a.tracker, imgsz=a.imgsz, stride=a.stride,
                run_seconds=round(elapsed, 1))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)

    final_path = os.path.join(a.out, "annotated.mp4")
    try:  # convert so browsers / Streamlit can play it
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
                        "-vcodec", "libx264", final_path], check=True)
        print(f"\nannotated video: {final_path}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\nFFmpeg not available: convert manually with:")
        print(f"ffmpeg -y -i {raw_path} -vcodec libx264 {final_path}")

    print(f"Run time: {elapsed:.0f} s | unique IDs: {df['id'].nunique() if len(df) else 0} | crops saved: {len(best)}")


if __name__ == "__main__":
    main()