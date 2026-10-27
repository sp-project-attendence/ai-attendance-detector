# Progress log

## Done
- Step A: synthetic session generation and presence/report foundation.
- M1: consent-aware YuNet/SFace enrolment into SQLite; user verified a real embedding was stored.
- M2: classroom identity pipeline and structured exports; user verified end-to-end processing. Recognition accuracy remains unvalidated because the current database has only a synthetic/test identity.
- M3: per-student presence reports and timelines; synthetic smoke test passes.
- M4: dashboard now starts from a local class video, runs existing `process_session.py` then `presence.py`, and loads reports/evidence internally. Teacher UI no longer asks for CSV/JSON files or exposes backend paths.
- M4 includes session status/summary, manual attendance entry in the dashboard, reconciliation, detail/timeline/confidence/evidence, teacher review actions, unknown-track assignment, audit log, and final attendance export.
- Dashboard stays white/light. `src/dashboard.py` syntax compilation passed; no new dependencies were added.
- Diagnosed the failed `class_1.mp4` dashboard run from its internal processing log: M2 was given the default `data\attendance.db`, but that file was not present at that location. The dashboard now supports the admin-only `ATTENDANCE_DB_PATH` environment setting and checks it before launching processing, with a path-free setup message for the UI.
- Prepared M1 for private real enrollment: `src/enrol.py` reports missing student folders, unreadable/no-face photos, multiple-face photos (largest face used), and students with zero usable photos. It replaces a student's gallery on re-enrollment, inserts every usable photo embedding, and removes stored embeddings when consent is not `yes`. `src/enrol_smoke.py` checks embedding counts and consent withdrawal using synthetic inputs only.
- Added an isolated temporary-video demo path: the dashboard can run M2 with same-video demo enrollment, then M3, without opening the real enrollment database. Demo gallery/roster, sessions, processing outputs, reports, and sample manual roll call live under `output\demo_sessions\`; all demo screens and M3 summaries are labeled as same-video demo, not accuracy evaluation. Sample manual marks are clearly marked editable examples for reconciliation demonstrations.
- Fixed the first video demo export failure: tracks with no nearby second-phase face sample (expected during the first 30% gallery-building segment) now use their aggregate track identity confidence as a fallback instead of calling `max()` on an empty sequence. Added a synthetic M2 regression assertion for both matched and missing nearby samples.
- Tuned the default tracker for a faster balanced run: YOLO11s, 960 px inference, every second frame, and 0.15 detector threshold. Face samples are capped at 12 per track and spaced at least 0.75 seconds apart. Identity matching thresholds are unchanged. The slower YOLO11m / 1280 px / every-frame profile can be selected in `config.yaml` if local testing shows it is needed.
- Dashboard session processing now streams tracker and face-sampling progress into the active status instead of hiding all backend output until M2 and M3 finish. The teacher sees only stage/count updates; raw backend logs remain internal for troubleshooting.
- Added setup-focused README instructions for cloning, installing, running the temporary video demo, optional consent-based enrollment, synthetic checks, confidence scoring, and Git privacy. Extended `.gitignore` for SQLite sidecars, ONNX weights, and additional video extensions.
- Fixed Streamlit's actual app theme, which the dashboard CSS alone could not override in its rendered dataframe. `.streamlit/config.toml` now forces the professional light palette and restrained green accent; restart Streamlit to apply it.
- Reworked the teacher-facing dashboard layout with an academic workspace header, light canvas, white summary cards, theme-independent readable reconciliation tables, restrained status chips, and clear Present/Absent actions. Overview now presents visible duration and session share; selected-student detail keeps the evidence timeline and review controls.
- Refreshed the dashboard with a light Lavender Haze Glass palette: soft lavender/periwinkle gradients, translucent rounded panels, coordinated table/status colors, polished buttons, and matching Plotly evidence/confidence colors. Updated the Streamlit light-theme accents; application behavior is unchanged.
- Spaced the existing Streamlit tabs into individually bordered, padded lavender buttons with a clear active state and responsive wrapping; expanded KPI, panel, chart, and form spacing without changing navigation or app behavior.
- Refined the dashboard toward a Premium SaaS presentation with compact accented overview metrics, tighter notice-to-navigation spacing, clearer alert/table surfaces, and responsive polished tabs; all dashboard workflows remain unchanged.
- Reworked the dashboard presentation into a violet application header, segmented existing tab rail, and balanced 3-by-2 overview KPI grid. Updated native tables, controls, notices, cards, and responsive styling without changing review or attendance behavior.
- Replaced the tab strip presentation with a dedicated branded sidebar and a selector for the same four dashboard destinations; existing section contents remain intact and render on selection. Applied a cohesive lavender application shell and redesigned cards, tables, controls, and responsive layout. Final attendance CSV generation still refreshes on every app rerun.
- Replaced the light lavender appearance with the Midnight Intelligence dark palette, styling the existing sidebar navigation, KPI cards, tables, notices, controls, session status, timeline, and confidence gauge. Existing navigation destinations and attendance/review/export behavior are retained.
- Separated the existing audit log and final CSV download into clearly labeled Audit Log and Export sidebar destinations; retained the previous audit navigation value for session-state compatibility and kept CSV generation on every rerun.
- Replaced the dark theme with the SIS-inspired slate-blue/white palette across the Streamlit theme, dashboard surfaces, metrics, tables, controls, status colors, and existing charts. The six metrics retain their values and are now wrapped with presentation-only keys for category accent styling. Syntax, TOML, Plotly gauge construction, and synthetic M3 checks pass. To run it, use `.\venv\Scripts\python.exe -m streamlit run src\dashboard.py` from the project root.

## Next
- README setup documentation is now present. The M5 accuracy evaluation system remains deferred per the user's instruction not to build an evaluation system yet.
- The local video-to-dashboard workflow has not been run end-to-end by the agent. Local camera capture, automatic session delivery, and ERP integration remain outside this demo; recognition accuracy also remains unvalidated.
- The actual M1 database location is not configured in this workspace. Set `ATTENDANCE_DB_PATH` to the SQLite database created by M1 before starting a new class session.
- Real enrollment is opt-in and has not been run here. Place the roster at `data\students.csv` and photos under `data\enrol\<student_id>\`; all of `data\` and `*.db` are ignored by Git. Review consent and local retention requirements before running it.
- The user-run video demo produced 22 temporary identities and completed tracking/gallery construction, but the first attempt then failed during M2 export; the empty-sample case is now fixed and synthetic checks pass. The complete video → M3 → dashboard rerun remains unverified here. Demo identities are same-video labels only; no accuracy claim is supported.
- The faster balanced settings have not been run against the classroom video, so actual latency and tracking coverage remain unmeasured. Ultralytics may need to fetch `yolo11s.pt` if it is not cached. This change is a speed/coverage compromise, not an accuracy evaluation.
- The slate-blue dashboard has passed syntax/TOML and synthetic checks but has not been visually checked in a browser. Restart the Streamlit server from the project root to load the updated theme config and page styling.

## How to run (Windows PowerShell)
Place `class_1.mp4` in the configured local demo video directory (`E:\zsp_project\attendance-data` by default), then launch from the project root:
```powershell
$env:ATTENDANCE_DB_PATH = 'C:\path\to\your\M1\attendance.db'
.\venv\Scripts\python.exe -m streamlit run src\dashboard.py
```

Replace the database example with the existing M1 SQLite database location. The app defaults to `data\attendance.db` when the environment variable is not set.

### Private M1 enrollment setup (Windows PowerShell)
Expected private files (never commit these):
```text
data\
  students.csv
  enrol\
    S001\photo_01.jpg
    S001\photo_02.jpg
    S002\photo_01.jpg
  attendance.db   # created by the command below
```

Roster header and example row:
```csv
student_id,name,roll_no,section,consent
S001,Example Student,R001,CSE-A,yes
```
Use `yes` only when recorded consent is present; other values are skipped. Supported photos are `.jpg`, `.jpeg`, and `.png`. The script uses the largest detected face if a photo contains multiple faces and prints that warning. It reports unreadable photos, photos with no face, missing folders, and students with no usable photos.

From the project root, build or refresh the local database explicitly (this command is not run automatically):
```powershell
.\venv\Scripts\python.exe src\enrol.py --students data\students.csv --photos data\enrol --db data\attendance.db
```
The default model files are `E:\zsp_project\models\yunet.onnx` and `E:\zsp_project\models\sface.onnx`; keep them outside Git. Set the dashboard to the same database before launch:
```powershell
$env:ATTENDANCE_DB_PATH = (Resolve-Path data\attendance.db).Path
.\venv\Scripts\python.exe -m streamlit run src\dashboard.py
```

To verify counts without displaying embeddings, use Python's SQLite aggregate queries only:
```powershell
.\venv\Scripts\python.exe -c 'import sqlite3; c=sqlite3.connect("data/attendance.db"); print(c.execute("SELECT student_id, COUNT(*) FROM enrolment_embeddings GROUP BY student_id ORDER BY student_id").fetchall()); print(c.execute("SELECT COUNT(*) FROM students s JOIN enrolment_embeddings e USING (student_id) WHERE lower(trim(s.consent)) <> ?", ("yes",)).fetchone())'
```
The second query should return `0`. This verifies stored row counts and consent filtering without selecting biometric blobs. Synthetic workflow check:
```powershell
.\venv\Scripts\python.exe src\enrol_smoke.py
```

In the dashboard, enter the class/section (for example CSE-A), select `class_1.mp4` from the classroom video list, and choose **Start class session**. The app runs M2 identity processing and then M3 presence/report generation. On completion, the teacher sees the session overview and review tabs. Enter or edit manual attendance in the app; it is not uploaded as a CSV. Processing artifacts and review state are stored under ignored `output\class_sessions\` for the local demo.

### Temporary same-video college demo (no real M1 database required)
Set the local video library and launch:
```powershell
$env:ATTENDANCE_VIDEO_DIR = 'E:\zsp_project\attendance-data'
.\venv\Scripts\python.exe -m streamlit run src\dashboard.py
```
In **Class Session**, enable **Temporary demo identities from this video**, choose `class_1.mp4` or `class_6.mp4`, and start the session. M2 tracks the full video, uses early face samples from the first 30% to build a per-session temporary gallery, then matches samples from the remaining video. M3 generates presence and visibility gaps; the dashboard pre-fills clearly labeled sample manual marks so the reconciliation categories can be demonstrated and edited. Demo artifacts and the demo SQLite gallery are kept under `output\demo_sessions\`, separate from the real M1 database. This is same-video enrollment and is not an accuracy evaluation. For identity matches in the later portion, the temporary gallery needs usable detected faces in the first 30%, and later tracks need enough face samples.

Synthetic demo-gallery helper check (no video or face models):
```powershell
.\venv\Scripts\python.exe src\demo_smoke.py
```

The camera tracker defaults to `yolo11s.pt`, 960 px, and every second frame to reduce session time while retaining a stronger detector than the old `yolo11n.pt` / 640 px / every-fifth-frame profile. A first run may download `yolo11s.pt` into Ultralytics' local cache. For slower high-coverage experiments, change `tracker_model` to `yolo11m.pt`, `tracker_image_size` to `1280`, and `tracker_stride` to `1`. Compare annotated output and internal track counts locally; identity thresholds are not lowered to force matches.

The video library defaults to `E:\zsp_project\attendance-data`; an administrator can configure another local folder with the `ATTENDANCE_VIDEO_DIR` environment variable. The dashboard shows video names, not filesystem paths. The selected video and local backend execution are real for the demo, while camera/ERP session provisioning is not connected. The current test identity does not establish recognition accuracy.
