"""Step 1-3: video -> YOLO person detection -> ByteTrack/BoT-SORT IDs -> tracks.csv + annotated video.

Usage:
  python track.py --video lecture.mp4
  python track.py --video lecture.mp4 --tracker botsort.yaml --stride 5
"""
import argparse, json, os
import cv2
import pandas as pd
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolo11m.pt")        # yolo11s.pt if you have no GPU
    ap.add_argument("--tracker", default="bytetrack.yaml")  # or botsort.yaml
    ap.add_argument("--imgsz", type=int, default=1280)      # classroom people are small -> high res
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=3)        # process every Nth frame (speed)
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(os.path.join(a.out, "annotated_raw.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (w, h))
    model = YOLO(a.model)
    rows = []
    results = model.track(source=a.video, stream=True, persist=True, tracker=a.tracker,
                          classes=[0], conf=a.conf, imgsz=a.imgsz,
                          vid_stride=a.stride, verbose=False)
    for i, r in enumerate(results):
        t = (i * a.stride) / fps
        if r.boxes.id is not None:
            for tid, box in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
                rows.append(dict(t=round(t, 3), id=tid, x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
        writer.write(r.plot())  # frame with boxes + track IDs drawn
        if i % 50 == 0:
            print(f"processed {i * a.stride}/{n_frames} frames", end="\r")
    writer.release()

    pd.DataFrame(rows).to_csv(os.path.join(a.out, "tracks.csv"), index=False)
    meta = dict(fps=fps, duration=n_frames / fps, step=a.stride / fps)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    print("\nDone. Now convert the video for browser playback:")
    print(f"ffmpeg -y -i {a.out}/annotated_raw.mp4 -vcodec libx264 {a.out}/annotated.mp4")


if __name__ == "__main__":
    main()
