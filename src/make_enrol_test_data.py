"""Create a tiny all-synthetic M1 smoke-test roster and blank images."""
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    root = Path("output/enrol_test")
    (root / "photos" / "SYN001").mkdir(parents=True, exist_ok=True)
    (root / "photos" / "SYN002").mkdir(parents=True, exist_ok=True)
    with (root / "students.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "name", "roll_no", "section", "consent"])
        writer.writerow(["SYN001", "Synthetic Consent Yes", "T001", "TEST", "yes"])
        writer.writerow(["SYN002", "Synthetic Consent No", "T002", "TEST", "no"])
    blank = np.full((240, 320, 3), 245, dtype=np.uint8)
    cv2.imwrite(str(root / "photos" / "SYN001" / "blank.png"), blank)
    cv2.imwrite(str(root / "photos" / "SYN002" / "ignored.png"), blank)
    print(f"Synthetic M1 test data created at {root}")
    print("Expected: SYN001 reports no face; SYN002 is skipped due to consent.")


if __name__ == "__main__":
    main()
