"""Synthetic M2 logic smoke check; no video, models, photos, or database needed."""
from process_session import (aggregate_identity, identity_stitches,
                             observation_similarity, resolve_overlapping_assignments)


def main():
    # Scores stand in for cosine similarities from several fake face frames.
    match = aggregate_identity([
        ({"SYN001": 0.81, "SYN002": 0.20}, 1.0),
        ({"SYN001": 0.78, "SYN002": 0.22}, 0.8),
        ({"SYN001": 0.83, "SYN002": 0.18}, 0.9),
    ], threshold=0.363, margin=0.05, min_samples=2)
    assert match["student_id"] == "SYN001", match
    too_few = aggregate_identity([({"SYN001": 0.99}, 1.0)], 0.363, 0.05, 2)
    assert too_few["student_id"] is None and too_few["reason"] == "too_few_face_samples", too_few
    assert observation_similarity([], "SYN001", 0.82) == 0.82
    assert observation_similarity([{"similarities": {"SYN001": 0.91}}], "SYN001", 0.82) == 0.91
    weak_margin = aggregate_identity([
        ({"SYN001": 0.70, "SYN002": 0.68}, 1.0),
        ({"SYN001": 0.71, "SYN002": 0.69}, 1.0),
    ], 0.363, 0.05, 2)
    assert weak_margin["student_id"] is None and weak_margin["reason"] == "below_runner_up_margin", weak_margin

    rows = [
        {"track_id": 10, "student_id": "SYN001", "identity_confidence": 0.82,
         "first_seen_sec": 1.0, "last_seen_sec": 12.0, "status": "identified", "reason": "matched"},
        {"track_id": 11, "student_id": "SYN001", "identity_confidence": 0.74,
         "first_seen_sec": 8.0, "last_seen_sec": 17.0, "status": "identified", "reason": "matched"},
        {"track_id": 12, "student_id": "SYN001", "identity_confidence": 0.78,
         "first_seen_sec": 20.0, "last_seen_sec": 28.0, "status": "identified", "reason": "matched"},
    ]
    resolved, conflicts = resolve_overlapping_assignments(rows)
    assert resolved[0]["student_id"] == "SYN001"
    assert resolved[1]["student_id"] is None and len(conflicts) == 1
    groups, merges = identity_stitches(resolved, max_gap=8.0)
    assert len(groups) == 1 and len(merges) == 1, (groups, merges)
    print("M2 synthetic smoke check passed: multi-frame vote, abstention, conflict, identity stitching")


if __name__ == "__main__":
    main()
