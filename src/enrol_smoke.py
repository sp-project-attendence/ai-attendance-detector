"""Synthetic check for consent filtering and storing every usable embedding."""
import csv
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import enrol


class FakeDetector:
    def setInputSize(self, size):
        pass

    def detect(self, image):
        # Two faces verify that enrolment records the safe largest-face choice.
        return 0, np.array([
            [0, 0, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)


class FakeRecognizer:
    def alignCrop(self, image, face):
        return image

    def feature(self, aligned):
        return np.ones((1, 8), dtype=np.float32)


def main():
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix=".enrol-smoke-", dir=project_root) as temporary:
        root = Path(temporary)
        roster = root / "students.csv"
        photos = root / "enrol"
        (photos / "S001").mkdir(parents=True)
        (photos / "S002").mkdir(parents=True)
        for name in ("one.jpg", "two.jpg"):
            (photos / "S001" / name).write_bytes(b"synthetic placeholder")
        (photos / "S002" / "ignored.jpg").write_bytes(b"synthetic placeholder")
        with roster.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["student_id", "name", "roll_no", "section", "consent"])
            writer.writerow(["S001", "Synthetic One", "R001", "A", "yes"])
            writer.writerow(["S002", "Synthetic Two", "R002", "A", "no"])

        with patch.object(enrol, "make_models", return_value=(FakeDetector(), FakeRecognizer())), \
             patch.object(enrol.cv2, "imread", return_value=np.zeros((32, 32, 3), dtype=np.uint8)):
            enrol.enrol(roster, photos, root / "attendance.db", root / "yunet.onnx", root / "sface.onnx")

        connection = sqlite3.connect(root / "attendance.db")
        try:
            consenting_count = connection.execute(
                "SELECT COUNT(*) FROM enrolment_embeddings WHERE student_id='S001'"
            ).fetchone()[0]
            declined_count = connection.execute(
                "SELECT COUNT(*) FROM enrolment_embeddings WHERE student_id='S002'"
            ).fetchone()[0]
        finally:
            connection.close()
        assert consenting_count == 2, f"expected 2 embeddings, found {consenting_count}"
        assert declined_count == 0, f"declined student unexpectedly has {declined_count} embeddings"

        # Simulate consent being withdrawn after a previously successful enrollment.
        with roster.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["student_id", "name", "roll_no", "section", "consent"])
            writer.writerow(["S001", "Synthetic One", "R001", "A", "no"])
            writer.writerow(["S002", "Synthetic Two", "R002", "A", "no"])
        enrol.enrol(roster, photos, root / "attendance.db", root / "yunet.onnx", root / "sface.onnx")
        connection = sqlite3.connect(root / "attendance.db")
        try:
            remaining = connection.execute("SELECT COUNT(*) FROM enrolment_embeddings").fetchone()[0]
        finally:
            connection.close()
        assert remaining == 0, f"consent withdrawal left {remaining} embeddings"
        print("PASS: all usable embeddings stored; declined/withdrawn consent has no embeddings")


if __name__ == "__main__":
    main()
