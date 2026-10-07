"""Create a small, deterministic synthetic session for Step A."""
import csv
import random
from pathlib import Path

OUT = Path("output/simulated")
DURATION = 600
STEP = 5

STUDENTS = [
    ("S001", "Asha Rao", "01", 0, 600, 0.91),
    ("S002", "Dev Shah", "02", 75, 600, 0.86),       # late first seen
    ("S003", "Mira Das", "03", 0, 510, 0.88),        # early last seen
    ("S004", "Noah Iyer", "04", 0, 600, 0.46),       # low confidence
    ("S005", "Ira Sen", "05", 0, 600, 0.90),
    ("S006", "Zoya Khan", "06", 0, 600, 0.84),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(17)
    with (OUT / "students.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["student_id", "name", "roll_no"])
        w.writerows((s[0], s[1], s[2]) for s in STUDENTS)
    with (OUT / "observations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["student_id", "t", "similarity", "face_detected"])
        for sid, _, _, start, end, confidence in STUDENTS:
            for t in range(start, end, STEP):
                if sid == "S005" and 180 <= t < 230 or sid == "S006" and (120 <= t < 155 or 330 <= t < 365 or 470 <= t < 505):
                    continue
                face = rng.random() > (0.28 if sid == "S004" else 0.08)
                w.writerow([sid, t, round(max(0.1, min(0.99, rng.gauss(confidence, 0.025))), 3), int(face)])
    # S006 is deliberately marked present despite no simulated visibility.
    with (OUT / "manual_attendance.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["student_id", "manual_status"])
        w.writerows((sid, status) for sid, status in [("S001", "present"), ("S002", "present"), ("S003", "present"), ("S004", "present"), ("S005", "absent"), ("S006", "present")])
    print(f"Synthetic session created: {OUT} ({DURATION} seconds, {len(STUDENTS)} students)")


if __name__ == "__main__":
    main()
