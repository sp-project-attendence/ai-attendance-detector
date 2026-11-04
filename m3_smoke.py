"""Build and check a synthetic M3 report without using M1/M2 private data."""
import csv
import json
from pathlib import Path

from presence import generate_report


def main():
    root = Path("output/m3_smoke")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "students.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "name", "consent"])
        writer.writerows([
            ["SYN001", "Synthetic Regular", "yes"],
            ["SYN002", "Synthetic Gaps", "yes"],
            ["SYN003", "Synthetic Not Seen", "yes"],
        ])
    with (root / "presence_input.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "track_id", "t", "similarity", "face_detected"])
        for t in (0, 5, 10, 25, 30):
            writer.writerow(["SYN001", 1, t, 0.90, 1])
        for t in (90, 95, 100, 200, 205, 300, 305):
            detected = int(t == 90)
            writer.writerow(["SYN002", 2, t, 0.40 if detected else "", detected])
    (root / "meta.json").write_text(json.dumps({"duration": 600, "step": 5}), encoding="utf-8")
    session_path = root / "session.json"
    session_path.write_text(json.dumps({
        "scheduled_start": "09:00:00", "scheduled_end": "09:10:00",
        "video_start": "09:00:00",
    }), encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    report, _ = generate_report(root, root, students_csv=root / "students.csv",
        session_json=session_path, config_path=project_root / "config.yaml")
    by_id = report.set_index("student_id")
    assert json.loads(by_id.loc["SYN001", "visibility_intervals"]) == [[0.0, 35.0]]
    assert by_id.loc["SYN001", "confidence"] == "High"
    assert "Early last seen" in by_id.loc["SYN001", "flags"]
    assert by_id.loc["SYN002", "gap_count"] == 2
    assert by_id.loc["SYN002", "status"] == "Seen (low confidence)"
    assert "Late first seen" in by_id.loc["SYN002", "flags"]
    assert "Long gap" in by_id.loc["SYN002", "flags"]
    assert by_id.loc["SYN003", "status"] == "Not seen"
    summary = (root / "session_summary.md").read_text(encoding="utf-8").lower()
    assert "does not mean the student left the room" in summary
    relative_report, _ = generate_report(root, root, students_csv=root / "students.csv",
        duration_sec=600, config_path=project_root / "config.yaml")
    relative_gaps = relative_report.set_index("student_id")
    assert "Early last seen" in relative_gaps.loc["SYN001", "flags"]
    demo_report, demo_timelines = generate_report(root, root, students_csv=root / "students.csv",
        session_json=session_path, config_path=project_root / "config.yaml", demo=True)
    assert demo_report["demo_mode"].all()
    assert "not an accuracy evaluation" in demo_report.iloc[0]["demo_label"].lower()
    assert all(item["demo_mode"] for item in demo_timelines.values())
    assert "same-video enrollment" in (root / "session_summary.md").read_text(encoding="utf-8").lower()
    print("M3 synthetic smoke check passed; report artifacts are in output/m3_smoke")


if __name__ == "__main__":
    main()
