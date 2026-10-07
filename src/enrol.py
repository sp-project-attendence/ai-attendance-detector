"""Enrol consenting students from photos using OpenCV YuNet and SFace."""
import argparse
import csv
import sqlite3
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def make_models(detector_path, recognizer_path, score_threshold=0.6):
    """Load the OpenCV Zoo ONNX models; model files stay outside this repo."""
    for model_path in (detector_path, recognizer_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if model_path.stat().st_size < 10_000:
            raise ValueError(f"{model_path} is too small to be a model file")
    detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (320, 320), score_threshold, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    return detector, recognizer


def largest_face(detector, image):
    """Return the largest detected face row, or None when no face is found."""
    face, _ = largest_face_and_count(detector, image)
    return face


def largest_face_and_count(detector, image):
    """Return the largest face and the number of faces detected."""
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None, 0
    return max(faces, key=lambda face: float(face[2]) * float(face[3])), len(faces)


def make_embedding(recognizer, image, face):
    aligned = recognizer.alignCrop(image, face)
    vector = recognizer.feature(aligned).flatten().astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if not norm:
        raise ValueError("SFace returned an empty embedding")
    return vector / norm


def initialize_database(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            roll_no TEXT NOT NULL,
            section TEXT NOT NULL,
            consent TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS enrolment_embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL REFERENCES students(student_id),
            photo_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)


def read_students(csv_path):
    required = {"student_id", "name", "roll_no", "section", "consent"}
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("students.csv is missing columns: " + ", ".join(sorted(missing)))
        return list(reader)


def process_student(row, photo_folder, detector, recognizer):
    """Build one embedding per usable image and return simple per-photo outcomes."""
    used, failures, warnings = [], [], []
    folder_exists = photo_folder.is_dir()
    images = sorted(p for p in photo_folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if folder_exists else []
    for photo in images:
        image = cv2.imread(str(photo))
        if image is None:
            failures.append((photo.name, "could not read image"))
            continue
        # Resize oversized photos while preserving aspect ratio.
        scale = min(1.0, 1280.0 / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        face, face_count = largest_face_and_count(detector, image)
        if face is None:
            failures.append((photo.name, "no face detected"))
            continue
        if face_count > 1:
            warnings.append((photo.name, f"{face_count} faces detected; largest face used"))
        used.append((photo.name, make_embedding(recognizer, image, face)))
    return used, failures, warnings, len(images), folder_exists


def enrol(students_csv, photos_root, database, detector_path, recognizer_path):
    rows = read_students(students_csv)
    consenting_rows = [row for row in rows if row["consent"].strip().lower() == "yes"]
    # Do not require model files when every roster entry has declined consent.
    models = make_models(detector_path, recognizer_path) if consenting_rows else (None, None)
    detector, recognizer = models
    database.parent.mkdir(parents=True, exist_ok=True)
    # sqlite3's context manager commits/rolls back but does not close the handle.
    with closing(sqlite3.connect(database)) as connection, connection:
        initialize_database(connection)
        for row in rows:
            sid = row["student_id"].strip()
            if not sid:
                print("Skipped roster row with empty student_id")
                continue
            connection.execute("""INSERT INTO students(student_id,name,roll_no,section,consent)
                VALUES(?,?,?,?,?) ON CONFLICT(student_id) DO UPDATE SET
                name=excluded.name, roll_no=excluded.roll_no,
                section=excluded.section, consent=excluded.consent""",
                (sid, row["name"].strip(), row["roll_no"].strip(),
                 row["section"].strip(), row["consent"].strip().lower()))
            if row["consent"].strip().lower() != "yes":
                # A later consent withdrawal must not leave an older face gallery active.
                connection.execute("DELETE FROM enrolment_embeddings WHERE student_id=?", (sid,))
                print(f"{sid} ({row['name']}): skipped — consent is not yes; 0 photos used")
                continue
            used, failures, warnings, _, folder_exists = process_student(
                row, photos_root / sid, detector, recognizer
            )
            # Replace this student's old gallery only after scanning this run.
            connection.execute("DELETE FROM enrolment_embeddings WHERE student_id=?", (sid,))
            connection.executemany(
                "INSERT INTO enrolment_embeddings(student_id,photo_name,embedding) VALUES(?,?,?)",
                [(sid, name, vector.astype(np.float32).tobytes()) for name, vector in used],
            )
            print(f"{sid} ({row['name']}): {len(used)} photos used")
            if not folder_exists:
                print(f"  missing student folder: {photos_root / sid}")
            for name, reason in failures:
                print(f"  failed: {name} — {reason}")
            for name, reason in warnings:
                print(f"  warning: {name} — {reason}")
            if len(used) == 0:
                print("  warning: zero usable enrollment photos; student has no face embeddings")
        connection.commit()
    print(f"Database updated: {database}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    external_models = project_root.parent / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=Path, default=project_root / "data" / "students.csv")
    parser.add_argument("--photos", type=Path, default=project_root / "data" / "enrol")
    parser.add_argument("--db", type=Path, default=project_root / "data" / "attendance.db")
    parser.add_argument("--det", type=Path, default=external_models / "yunet.onnx")
    parser.add_argument("--rec", type=Path, default=external_models / "sface.onnx")
    args = parser.parse_args()
    if not args.students.is_file():
        parser.error(f"students CSV not found: {args.students}")
    enrol(args.students, args.photos, args.db, args.det, args.rec)


if __name__ == "__main__":
    main()
