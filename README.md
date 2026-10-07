# AI Attendance Anomaly Detector

A local college demo for reviewing classroom visibility evidence alongside teacher-recorded attendance. The system processes a selected classroom video, estimates student identity and visible time, reports confidence and visibility anomalies, and presents the results for teacher review.

**The teacher remains the attendance authority.** Video evidence does not mark a student present or absent automatically. Not being visible does **not** mean that a student left the room. This project does not analyze attention or behavior and does not connect to an ERP.

## Current demo limits

- The temporary demo mode derives its identities from the first 30% of the same video, then matches the remainder. This is only to demonstrate the application flow; it is **not an accuracy evaluation**.
- Back-row, occluded, turned-away, or masked faces may be missed. There is no liveness detection, so photo spoofing is not prevented. Recognition systems may have demographic performance differences.
- Recognition accuracy and tracking coverage have not been validated on a labelled, independent test set. Do not use demo outputs for real attendance decisions.
- This is a local demonstration, not a production deployment. It has no institutional login or ERP integration. Obtain recorded consent and review applicable privacy requirements, including India's DPDP Act 2023, before any real deployment.

## Requirements

- Windows 10/11 and PowerShell
- Python 3.10 or newer (Python 3.11 is a practical choice for package compatibility)
- The packages listed in `requirements.txt`
- For video sessions: YuNet and SFace ONNX models from the OpenCV Zoo, and an internet connection on the first YOLO run if `yolo11s.pt` is not already cached

Do not put videos, student photos, face embeddings, databases, model weights, or virtual environments in Git. Keep them in the ignored local folders or outside the repository as described below.

## Clone and install

Open PowerShell and choose a location for the project. The following commands clone the GitHub repository and create a local virtual environment:

```powershell
git clone https://github.com/mitish549/ai-attendance-detector.git
Set-Location ai-attendance-detector
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.10+ and use its version with the Python launcher. The `venv` folder is local and ignored by Git. PyTorch is installed as a dependency of Ultralytics; GPU/CUDA setup depends on the computer and is optional.

## Local model files

Download `face_detection_yunet_2023mar.onnx` from the [YuNet model folder](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and `face_recognition_sface_2021dec.onnx` from the [SFace model folder](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) in the official OpenCV Zoo. Rename the downloaded files to `yunet.onnx` and `sface.onnx`, respectively. Keep them outside the cloned repository. The default expected layout is a `models` folder beside the project folder:

```text
your-workspace\
  ai-attendance-detector\
  models\
    yunet.onnx
    sface.onnx
```

For example, if the project is at `E:\zsp_project\ai-attendance-detector`, the model files go in `E:\zsp_project\models`. Do not commit these model files. YOLO weights are managed by Ultralytics in its local cache and are also not part of this repository.

## Run the temporary video demo

Put a classroom video in a private local folder outside the repository. Set that folder for the dashboard, then launch Streamlit from the project root:

```powershell
$env:ATTENDANCE_VIDEO_DIR = 'E:\zsp_project\attendance-data'
Set-Location E:\zsp_project\ai-attendance-detector
.\venv\Scripts\python.exe -m streamlit run src\dashboard.py
```

In the dashboard:

1. Enter the class or section and teacher name.
2. Enable **Temporary demo identities from this video**.
3. Select a video and choose **Start class session**.
4. Review the generated visibility evidence and edit the clearly labelled sample manual attendance marks for the demonstration.

The app runs tracking/identity processing (M2), then presence/report generation (M3). Demo reports, evidence crops, temporary identities, and sample roll-call marks stay under `output\demo_sessions\` and are not real enrollment records. Processing can take a while, especially on the first run while YOLO weights are downloaded. The default profile is YOLO11s at 960 px on every second frame; this balances speed and coverage but can miss some small or distant people.

The dashboard can also accept a different local video folder. Set `ATTENDANCE_VIDEO_DIR` before launching; the teacher UI displays video names, not local filesystem paths.

## Optional: private real enrollment

Only use this workflow where students have provided recorded consent and local data handling is approved. Keep all roster, image, and database files in the ignored `data` folder:

```text
data\
  students.csv
  enrol\
    S001\photo_01.jpg
    S001\photo_02.jpg
    S002\photo_01.jpg
  attendance.db
```

The roster must have this header and one row per student:

```csv
student_id,name,roll_no,section,consent
S001,Example Student,R001,CSE-A,yes
```

Use `yes` only when consent is recorded. Other consent values are skipped. Supported photo types are JPG, JPEG, and PNG. The enrollment script stores one embedding per usable photo, reports missing folders and unusable photos, and uses the largest face if a photo contains multiple faces.

Build the database explicitly; the dashboard never creates or refreshes real enrollment data automatically:

```powershell
Set-Location E:\zsp_project\ai-attendance-detector
.\venv\Scripts\python.exe src\enrol.py --students data\students.csv --photos data\enrol --db data\attendance.db
$env:ATTENDANCE_DB_PATH = (Resolve-Path data\attendance.db).Path
$env:ATTENDANCE_VIDEO_DIR = 'E:\zsp_project\attendance-data'
.\venv\Scripts\python.exe -m streamlit run src\dashboard.py
```

In the dashboard, leave temporary demo identities disabled to use the M1 database. Never use sample same-video demo identities as a replacement for real consent-based enrollment.

## Synthetic checks

These checks use generated/synthetic inputs; they do not process classroom video or require real student data:

```powershell
Set-Location E:\zsp_project\ai-attendance-detector
.\venv\Scripts\python.exe src\m2_smoke.py
.\venv\Scripts\python.exe src\m3_smoke.py
.\venv\Scripts\python.exe src\demo_smoke.py
.\venv\Scripts\python.exe src\enrol_smoke.py
```

The M3 confidence score is:

```text
100 × (0.50 × mean recognition similarity
     + 0.30 × fraction of visible time with a detectable face
     + 0.20 × min(face sample count / 5, 1))
```

Inputs are clamped to 0–1. Confidence labels are High at 75+, Medium at 55–74.9, and Low below 55. This score describes evidence quality; it is not a guarantee of identity correctness.

## Privacy and GitHub checklist

Before staging or pushing code, check the Git status and staged filenames. These must remain local/private and must not appear in a commit:

- `data\` including roster, enrollment photos, and SQLite databases
- `output\` including evidence crops, session reports, and temporary demo databases
- `models\` and all model weights
- videos, embeddings, and the `venv\` environment
- `.env` files or credentials

The repository `.gitignore` excludes the standard local data, output, model, database, video, and environment paths. Still review what Git will publish before each push:

```powershell
git status --short
git diff --cached --name-only
```

If private files were committed previously, adding them to `.gitignore` does not remove them from Git history. Remove them from the repository history and rotate any exposed secrets before sharing the repository.

## How it works

```text
Classroom video → YOLO person tracking → YuNet face detection → SFace embeddings
                → M2 identities → M3 visibility/confidence/flags → teacher review
```

CSV and JSON reports are internal processing artifacts. The teacher works from the dashboard. Visibility intervals, gaps, and early last-seen flags are evidence for review and do not prove entry, exit, or absence.
