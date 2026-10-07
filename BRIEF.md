# Project brief: AI Attendance Anomaly Detector (face-recognition version)

## 1. Goal
Help a teacher cross-verify manual attendance using classroom video. The system answers ONE question:

> "Was this enrolled student visible in the room, when, and how reliably could we tell?"

It does NOT measure attention, engagement or behaviour. It never marks attendance on its own.
The teacher confirms or corrects every record.

## 2. Workflow
1. **Registration:** collect each student's face photos and link them to the college record (student_id, name, roll_no).
2. **Class:** one fixed camera inside the classroom records the whole session.
3. **Processing:** detect and track people, find faces, recognise them against the enrolled students, build per-student visibility data.
4. **Report:** per-student time seen, first seen, last seen, gaps, confidence, flags.
5. **Teacher** takes manual attendance as usual, then opens the dashboard and compares manual vs video.
6. **Review:** teacher confirms matches, reviews mismatches and low-confidence cases, corrects or follows up. Every action is logged.

## 3. What already exists in the repo (reuse and refactor, do not rewrite from scratch)
- `src/track.py`: YOLO + ByteTrack/BoT-SORT. Writes `tracks.csv` (columns `t,id,x1,y1,x2,y2,conf`; `t` in seconds), `meta.json` (fps, duration, step, settings, run_seconds), `annotated.mp4`, `crops/`.
- `src/check_ids.py`: tracking-quality check (unique IDs, short-lived IDs).
- `src/stitch.py`: merges broken tracks by time gap, position and box size. Writes `tracks_stitched.csv` (adds `orig_id`) and `merges.csv`.
- `src/faces.py`: OpenCV YuNet + SFace. Commands `extract`, `enrol`, `match`. **Written but NOT yet tested with the real model files; expect bugs and fix them.**
- `src/analyze.py`, `src/dashboard.py`: earlier prototypes (track-ID based attendance table and Streamlit dashboard). Replace or refactor as needed.
- Environment: Windows, PowerShell, Python 3.14, venv in the repo folder (git-ignored). ultralytics, torch, opencv-python 5.x, pandas, streamlit, plotly, lap are installed.

## 4. Tech constraints
- Python, compatible with 3.10+. Keep new dependencies to a minimum and say why for each one.
- Faces: `cv2.FaceDetectorYN` (YuNet) and `cv2.FaceRecognizerSF` (SFace), ONNX models from the OpenCV Zoo placed in `../models/yunet.onnx` and `../models/sface.onnx` (outside the repo, never committed). Do NOT use dlib or InsightFace (install problems on Windows, non-commercial model licence).
- Storage: SQLite for students, enrolment embeddings, sessions, results and audit log. CSV/JSON exports for every report.
- Dashboard: Streamlit + Plotly.
- Config: all thresholds in one `config.yaml`, documented.
- Never commit videos, face images, embeddings, models, `.db` files or the venv. Add `data/`, `models/`, `*.db`, `*.npz` to `.gitignore` (it already ignores `*.mp4`, `*.avi`, `*.pt`, `output/`, `run*/`, `crops/`).

## 5. Data handling during development (important)
- The agent must NOT be given, and must not ask for, real student photos, real face embeddings or real classroom video of identifiable people.
- Develop and test with synthetic data only: fake `students.csv`, generated tracks, random unit-norm vectors as stand-in embeddings.
- Real enrolment photos and real videos stay on the user's machine in `data/` (git-ignored). The USER runs the real pipeline locally and pastes back errors.
- The face gallery is built ONLY from enrolment photos in `data/enrol/`, never from the session video being tested (that would give meaningless accuracy). Exception: demo mode (section 11), which must be labelled as such everywhere.
- From session video the system may create "unknown face groups" (similar faces among unmatched tracks) so the teacher can label them. Labelling adds enrolment samples only after the teacher confirms and consent is recorded.

## 6. Components

### 6.1 Enrolment (`src/enrol.py`)
- Input: `students.csv` (`student_id,name,roll_no,section,consent`) and photos in `data/enrol/<student_id>/*.jpg` (3 to 5 per student).
- Detect the largest face per photo, compute an SFace embedding, store ALL embeddings per student in SQLite (not only the mean).
- Skip students whose consent is not `yes`. Print photos used and photos where no face was found, per student.

### 6.2 Session processing (`src/process_session.py`)
Inputs: video path and `session.json` (see 6.5).
1. Run YOLO + tracker on sampled frames (stride configurable) to get tracks.
2. For each track take several sharp, large head crops spread over time, detect faces, compute embeddings.
3. Identity per track = VOTE over many frames (similarity-weighted), never a single frame:
   - the best match must exceed THRESHOLD and beat the runner-up by MARGIN, otherwise the track is "unknown";
   - starting THRESHOLD 0.363 (cosine, from OpenCV's demo; must be tuned on validation data), MARGIN 0.05, both in `config.yaml`.
4. Merge tracks that resolve to the same student (identity-based stitching), in addition to spatial stitching with `stitch.py`.
5. A student cannot be in two places at once: if two overlapping tracks claim the same student keep the higher-confidence one and flag the conflict.
6. Keep unknown tracks (best face crop + time range) for the teacher to label.

### 6.3 Presence report (`src/presence.py`)
For each enrolled student output:
- `first_seen`, `last_seen` (clock times), `total_visible_seconds`, visible % of the scheduled session;
- visibility intervals (merge gaps shorter than `GAP_TOL`, default 15 s), gap count, longest gap;
- confidence components: mean recognition similarity, % of visible time with a detectable face, number of face samples;
- `confidence_score` 0 to 100 (document the formula in the README) and a label High / Medium / Low;
- `status`: "Seen" | "Seen (low confidence)" | "Not seen".

Semantics: "not visible" does NOT mean "left the room" (hidden behind someone, head down, etc.). Use the words "first seen" and "last seen", never "entry" and "exit".

Flags (thresholds in config): late first seen, early last seen, multiple gaps, long gap, low visibility, low confidence.
Outputs: `report.csv`, a per-student timeline JSON, and a plain-text/markdown summary per student ("video to text").

### 6.4 Reconciliation and dashboard (`src/dashboard.py`)
- Teacher uploads or enters manual attendance (`student_id,present/absent`).
- Reconciliation table with categories:
  - Match (manual present and seen, or manual absent and not seen)
  - Possible proxy (manual present, NOT seen)
  - Seen but marked absent (late after marking, or teacher error)
  - Low confidence (needs review)
- Student detail view: time seen, first seen, last seen, gap timeline (Plotly), confidence meter with breakdown, best face evidence crops, flags with plain-language reasons.
- Unknown-faces panel: teacher assigns a name from the roster.
- Actions per student: Confirm, Correct (reason required), Follow-up (note). "Confirm all matches" button.
- Audit log (timestamp, teacher, student, old value, new value, reason). Export `final_attendance.csv`.
- Anything the teacher has not reviewed stays visibly "Pending".

### 6.5 Session timing
- Preferred: `session.json` with `class_id`, `section`, `scheduled_start`, `scheduled_end` (clock times) and `video_start` (clock time of the first frame). The teacher can edit it per session.
- Optional hint: read the video's `creation_time` metadata (ffprobe) and show it as a SUGGESTION, never as a fact.
- Fallback (demo): with no `session.json`, treat the whole video as the class (0:00 = start, video length = duration). Mark the report "relative times" and disable late/early flags that need real clock times.
- If one video covers several periods, split it using a timetable list of periods (start/end) and process each period as a separate session.

### 6.6 Evaluation (`src/evaluate.py`)
- Ground truth CSV per student: true visible intervals labelled by humans.
- Metrics: identification accuracy and unknown rate on labelled face samples, false-accept rate, attendance agreement (seen / not seen) per student, mean absolute error of visible time, flag precision and recall.
- Enrolment photos must be DIFFERENT images from the test video. Choose threshold/margin on a validation session, report on a separate test session.

## 7. Synthetic data (`src/simulate.py`)
Because the sample video is short and nobody enters or leaves, generate fake sessions with late arrivals, early departures, multiple gaps and low-confidence cases, plus a fake teacher manual-attendance CSV with a few deliberate proxies. Used for unit tests and for demonstrating the flags and the reconciliation screen. Always label simulated results as simulated.

## 8. Privacy and ethics (requirements, not extras)
- Faces and embeddings are biometric personal data. Enrol only students with recorded consent (consent flag in DB; consent note and retention date in the README).
- Keep embeddings; keep raw photos and video only as long as configured (`retention_days`, `delete_video_after_processing`).
- The dashboard shows only the logged-in teacher's class (simple local user field is acceptable for now). Log every view and action.
- No attention or behaviour analysis. No automatic penalties. The teacher always decides.
- Document limitations in the README: back-row, occluded or masked faces; no liveness detection (photo spoofing not prevented); possible demographic bias; DPDP Act 2023 compliance needed before real deployment.

## 9. Non-goals (for now)
ERP integration (provide clean CSV/JSON export only), entrance/exit cameras, multi-camera fusion, liveness detection, attention analysis.

## 10. Milestones with acceptance criteria
Each milestone must be runnable from the command line on Windows PowerShell, have tests where feasible, and commit no data or secrets.
- **M1 Enrolment:** `students.csv` + photos go into the DB; the script prints photos used and failed per student.
- **M2 Identity:** one session video becomes tracks, then per-track identities with confidence; unknown tracks saved; conflicts flagged.
- **M3 Report:** `report.csv` and per-student summary text; unit tests for gap merging, flags and confidence on synthetic tracks.
- **M4 Dashboard:** reconciliation, student detail, confidence meter, teacher actions, audit log, `final_attendance.csv`.
- **M5 Evaluation and README:** evaluation script, documented `config.yaml`, Windows run instructions, privacy section.

## 11. Demo mode (sample video, no registration data)
Context: development uses a downloaded sample classroom video of strangers. No real photos or college database exist.
Add `--demo` to the pipeline. In demo mode:
1. Run tracking and face extraction on the whole video.
2. Build a DEMO gallery from faces found in the first 30% of the video, one entry per track with good faces, labelled "Student 01", "Student 02", ... with fake roll numbers written to `demo_students.csv` (the user may rename them).
3. Recognise people in the remaining 70% against that gallery and produce the normal report, reconciliation and dashboard.
4. Treat the whole video as the class; mark the report "relative times, demo".
5. Label every demo result: **"DEMO: same-video enrolment, not an accuracy evaluation."**
Real enrolment (`data/enrol/<student_id>/*.jpg`) remains the primary path; demo mode is only a convenience switch.

## 12. Working agreement
- Work one milestone at a time, starting with M1. After each one, stop and summarise what changed, how to run it and what to test.
- Ask before assuming. Open questions:
  - source of scheduled class times (timetable file vs entered per session)
  - retention period for embeddings and video
  - whether the dashboard needs a real login (default: local user field + audit log)
- Keep the code simple and commented for a third-year student to read. Explain any new dependency.
- Write commands for Windows PowerShell.

## 13. Build order (usage is limited, so every stopping point must leave something that works)
- **Step A:** `simulate.py` + `presence.py` + dashboard reconciliation, using SIMULATED data only. No face models, no video. This is the first demo.
- **Step B:** M1 enrolment.
- **Step C:** M2 identity on the sample video (demo mode).
- **Step D:** M5 evaluation + README.
After each step: update `PROGRESS.md` (done, next, how to run) and give the exact PowerShell command to run it.
