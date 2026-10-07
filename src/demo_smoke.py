"""Synthetic check for the temporary demo gallery builder; does not process video."""
from pathlib import Path

import numpy as np

from process_session import build_demo_gallery, load_gallery


def main():
    output = Path(__file__).resolve().parents[1] / "output" / "demo_smoke"
    output.mkdir(parents=True, exist_ok=True)
    database = output / "demo_gallery.db"
    samples = {
        4: [{"t": 10.0, "det_score": 0.9, "sharpness": 80.0,
             "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
        8: [{"t": 20.0, "det_score": 0.8, "sharpness": 70.0,
             "embedding": np.array([0.0, 1.0, 0.0], dtype=np.float32)}],
        12: [{"t": 90.0, "det_score": 0.9, "sharpness": 90.0,
              "embedding": np.array([0.0, 0.0, 1.0], dtype=np.float32)}],
    }
    cutoff, count = build_demo_gallery(database, samples, duration_sec=100.0)
    gallery = load_gallery(database)
    assert cutoff == 30.0
    assert count == 2 and sorted(gallery) == ["DEMO001", "DEMO002"]
    assert all(vectors.shape == (1, 3) for vectors in gallery.values())
    assert (output / "demo_students.csv").is_file()
    print("PASS: demo gallery uses only early synthetic samples and stays under output/demo_smoke")


if __name__ == "__main__":
    main()
