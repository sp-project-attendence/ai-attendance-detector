"""WP2 track stitching: merge tracks that are probably the same person after a tracker break.

Idea: if track A ends, and track B starts a short time later, in about the same place and with
about the same box size, treat B as the continuation of A.  Seated students barely move, so
'same place' is a strong clue in a classroom.

Usage:
  python src/stitch.py --tracks <run>/tracks.csv
  python src/stitch.py --tracks <run>/tracks.csv --max_gap 8 --dist 0.5
Writes tracks_stitched.csv (id = merged id, orig_id = tracker's id) and merges.csv (what was merged).
"""
import argparse, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--max_gap", type=float, default=5.0, help="max seconds between end of A and start of B")
    ap.add_argument("--dist", type=float, default=0.5, help="max distance A-end to B-start, in box heights")
    ap.add_argument("--out", default=None, help="output folder (default: same folder as tracks.csv)")
    a = ap.parse_args()

    out = a.out or os.path.dirname(os.path.abspath(a.tracks))
    df = pd.read_csv(a.tracks).sort_values("t").reset_index(drop=True)
    df["cx"] = (df.x1 + df.x2) / 2
    df["cy"] = (df.y1 + df.y2) / 2
    df["h"] = df.y2 - df.y1

    info = {}
    for tid, d in df.groupby("id"):
        first, last = d.iloc[0], d.iloc[-1]
        info[tid] = dict(start=first.t, end=last.t, fx=first.cx, fy=first.cy,
                         lx=last.cx, ly=last.cy, h=float(d.h.median()))
    ids = sorted(info, key=lambda k: info[k]["start"])

    successor, merges = {}, []
    for b in ids:                       # b = the later track, look for its predecessor
        B = info[b]
        best = None
        for aid in ids:
            if aid == b or aid in successor:
                continue
            A = info[aid]
            gap = B["start"] - A["end"]
            if gap <= 0 or gap > a.max_gap:
                continue                 # must start after A ended, and not too long after
            dist = np.hypot(B["fx"] - A["lx"], B["fy"] - A["ly"]) / max(A["h"], 1)
            if dist > a.dist:
                continue
            if not 0.6 <= B["h"] / max(A["h"], 1) <= 1.6:
                continue
            cost = dist + gap / a.max_gap
            if best is None or cost < best[0]:
                best = (cost, aid, gap, dist)
        if best:
            successor[best[1]] = b
            merges.append(dict(from_id=best[1], to_id=b, gap_s=round(best[2], 2),
                               dist_box_heights=round(best[3], 2),
                               at_time=round(info[best[1]]["end"], 2)))

    mapping = {}
    for aid in ids:                      # ids are sorted by start time, so aid is processed before its successor
        if aid in successor:
            mapping[successor[aid]] = mapping.get(aid, aid)

    df["orig_id"] = df["id"]
    df["id"] = df["id"].map(lambda x: mapping.get(x, x))
    df = df.drop(columns=["cx", "cy", "h"])
    df.to_csv(os.path.join(out, "tracks_stitched.csv"), index=False)
    pd.DataFrame(merges).to_csv(os.path.join(out, "merges.csv"), index=False)

    life = lambda d: d.groupby("id")["t"].agg(lambda s: s.max() - s.min())
    before, after = pd.read_csv(a.tracks), df
    print(f"unique IDs: {before['id'].nunique()} -> {after['id'].nunique()}  ({len(merges)} merges)")
    print(f"IDs under 5 s: {(life(before) < 5).sum()} -> {(life(after) < 5).sum()}")
    for m in merges:
        print(f"  merged {m['from_id']} -> {m['to_id']} at {m['at_time']}s (gap {m['gap_s']}s, dist {m['dist_box_heights']})")


if __name__ == "__main__":
    main()
