"""Step 4-7: tracks -> presence intervals -> total time -> attendance % -> anomalies."""
from dataclasses import dataclass
import pandas as pd


@dataclass
class Params:
    gap_tol: float = 15.0        # sec: disappearance shorter than this = occlusion, still PRESENT
    min_track_sec: float = 30.0  # ignore tracks shorter than this (false detections / passers-by)
    late_frac: float = 0.10      # first seen after 10% of class -> late
    early_frac: float = 0.10     # last seen before final 10% -> early departure
    exits_flag: int = 3          # >= this many absences -> frequent exits
    present_pct: float = 75.0
    partial_pct: float = 40.0


def build_intervals(times, gap_tol, step):
    times = sorted(times)
    out, start, prev = [], times[0], times[0]
    for t in times[1:]:
        if t - prev > gap_tol:
            out.append((start, prev + step))
            start = t
        prev = t
    out.append((start, prev + step))
    return out


def analyze(tracks: pd.DataFrame, duration: float, step: float, p: Params):
    """Returns (summary DataFrame, {student: [(start, end), ...]})."""
    raw = {}
    for tid, g in tracks.groupby("id"):
        iv = build_intervals(g["t"].tolist(), p.gap_tol, step)
        if sum(e - s for s, e in iv) >= p.min_track_sec:
            raw[tid] = iv
    order = sorted(raw, key=lambda k: raw[k][0][0])  # simple demo mapping: by first appearance

    rows, intervals = [], {}
    for n, tid in enumerate(order, 1):
        name = f"Student {n:02d}"
        iv = raw[tid]
        intervals[name] = iv
        total = sum(e - s for s, e in iv)
        pct = min(100.0, total / duration * 100)
        exits = len(iv) - 1
        longest = max(e - s for s, e in iv)

        anomalies = []
        if iv[0][0] > p.late_frac * duration:
            anomalies.append("Late arrival")
        if iv[-1][1] < (1 - p.early_frac) * duration:
            anomalies.append("Early departure")
        if exits >= p.exits_flag:
            anomalies.append("Frequent exits")
        elif exits >= 2 and longest < 0.5 * total:
            anomalies.append("Irregular pattern")
        if pct < p.partial_pct:
            anomalies.append("Very low presence")

        status = "Present" if pct >= p.present_pct else "Partial" if pct >= p.partial_pct else "Absent"
        rows.append({"Student": name, "Track ID": tid, "Present (min)": round(total / 60, 1),
                     "Attendance %": round(pct, 1), "Status": status,
                     "Exits": exits, "Anomaly": ", ".join(anomalies) or "-"})
    return pd.DataFrame(rows), intervals
