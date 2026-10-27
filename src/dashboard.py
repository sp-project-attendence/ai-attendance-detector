"""Local class-session workflow and teacher attendance review dashboard."""
import csv
from html import escape
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = Path(os.environ.get("ATTENDANCE_VIDEO_DIR", r"E:\zsp_project\attendance-data"))
DATABASE_PATH = ROOT / "data" / "attendance.db"
if os.environ.get("ATTENDANCE_DB_PATH"):
    DATABASE_PATH = Path(os.environ["ATTENDANCE_DB_PATH"]).expanduser()
CONFIG_PATH = ROOT / "config.yaml"
SESSIONS_DIR = ROOT / "output" / "class_sessions"
SESSION_INDEX = SESSIONS_DIR / "sessions.json"
DEMO_SESSIONS_DIR = ROOT / "output" / "demo_sessions"
ATTENDANCE_CHOICES = ["Not recorded", "Present", "Absent"]


def safe_csv(path):
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def load_report(path):
    """Load an internal M3 report and return a clear validation message if invalid."""
    try:
        report = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None, "The session report is empty. Run processing for this session again."
    except pd.errors.ParserError:
        return None, "The session report is malformed. Ask an administrator to rerun session processing."
    except UnicodeDecodeError:
        return None, "The session report has an unsupported text encoding. Ask an administrator to rerun processing."
    except OSError:
        return None, "The session report is unavailable. Ask an administrator to rerun session processing."
    if report.empty:
        return None, "The session report is empty. Run processing for this session again."
    required = {"student_id", "name", "status"}
    missing = sorted(required - set(report.columns))
    if missing:
        return None, "The session report is missing required fields. Ask an administrator to check processing."
    return report, None


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sessions_index(demo=False):
    return DEMO_SESSIONS_DIR / "sessions.json" if demo else SESSION_INDEX


def load_sessions(demo=False):
    value = read_json(sessions_index(demo), [])
    return value if isinstance(value, list) else []


def save_sessions(sessions, demo=False):
    write_json(sessions_index(demo), sessions)


def relative_path(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def session_folder(session, key):
    return ROOT / session[key]


def discover_videos():
    """List file names only; the backend opens a video after a teacher starts a session."""
    if not VIDEO_DIR.is_dir():
        return []
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted((path for path in VIDEO_DIR.iterdir()
                   if path.is_file() and path.suffix.lower() in extensions), key=lambda p: p.name.lower())


def run_backend(video_path, class_id, teacher, demo=False, progress_callback=None):
    """Run the existing M2 and M3 scripts and store internal session metadata."""
    sessions_dir = DEMO_SESSIONS_DIR if demo else SESSIONS_DIR
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = sessions_dir / session_id
    m2_dir, m3_dir = run_dir / "processing", run_dir / "report"
    m2_dir.mkdir(parents=True, exist_ok=True)
    m3_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": session_id,
        "class_id": class_id.strip() or "Class session",
        "video_name": Path(video_path).name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "teacher": teacher,
        "demo_mode": bool(demo),
        "database_path": str(((m2_dir / "demo_gallery.db") if demo else DATABASE_PATH).resolve()),
        "status": "Running",
        "processing_dir": relative_path(m2_dir),
        "report_dir": relative_path(m3_dir),
        "error": "",
    }
    sessions = load_sessions(demo)
    sessions.insert(0, record)
    save_sessions(sessions, demo)

    log_path = run_dir / "processing.log"
    m2_command = [sys.executable, str(ROOT / "src" / "process_session.py"), "--video", str(video_path),
                  "--db", str(DATABASE_PATH), "--out", str(m2_dir), "--config", str(CONFIG_PATH)]
    m3_database = (m2_dir / "demo_gallery.db") if demo else DATABASE_PATH
    m3_command = [sys.executable, str(ROOT / "src" / "presence.py"), "--input", str(m2_dir),
                  "--db", str(m3_database), "--out", str(m3_dir), "--config", str(CONFIG_PATH)]
    if demo:
        m2_command.append("--demo")
        m3_command.append("--demo")
    commands = [m2_command, m3_command]
    with log_path.open("w", encoding="utf-8") as log_file:
        for label, command in zip(("M2 identity processing", "M3 presence report"), commands):
            log_file.write(f"[{label}]\n")
            log_file.flush()
            if progress_callback:
                progress_callback(label, "Starting video analysis" if label.startswith("M2")
                                  else "Building presence summary")
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            if process.stdout is not None:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    if progress_callback:
                        message = line.strip()
                        if message.startswith(("processed ", "face sampling:", "DEMO enrollment pass:",
                                               "DEMO recognition pass:", "Run time:")):
                            progress_callback(label, message)
                        elif message.startswith("Presence report:"):
                            progress_callback(label, "Presence summary generated")
            return_code = process.wait()
            if return_code != 0:
                log_file.write(f"\n{label} exited with code {return_code}\n")
                log_file.flush()
                record["status"] = "Failed"
                record["error"] = label
                save_sessions(sessions, demo)
                return record, False
    if demo:
        seed_demo_manual_scenario(m3_dir / "report.csv", m3_dir / "manual_attendance.csv")
        record["sample_manual_scenario"] = True
    record["status"] = "Completed"
    record["error"] = ""
    save_sessions(sessions, demo)
    return record, True


def load_roster(report, database_path=None):
    roster = report[[column for column in ("student_id", "name", "roll_no") if column in report]].copy()
    roster["student_id"] = roster.student_id.astype(str)
    if "roll_no" not in roster:
        roster["roll_no"] = ""
    database_path = Path(database_path) if database_path else DATABASE_PATH
    if database_path.is_file():
        try:
            with sqlite3.connect(database_path) as connection:
                extra = pd.read_sql_query(
                    "SELECT student_id, roll_no FROM students WHERE lower(trim(consent))='yes'", connection
                )
            extra["student_id"] = extra.student_id.astype(str)
            roster = roster.drop(columns=["roll_no"]).merge(extra, on="student_id", how="left")
        except (sqlite3.Error, pd.errors.DatabaseError):
            roster["roll_no"] = ""
    return roster.drop_duplicates("student_id")


def seed_demo_manual_scenario(report_path, destination):
    """Create clearly synthetic manual marks to demonstrate reconciliation categories."""
    report = pd.read_csv(report_path)
    marks = []
    first_seen_marked_absent = False
    for row in report.itertuples(index=False):
        if row.status == "Not seen":
            manual = "present"  # deliberate possible-proxy review case
        elif not first_seen_marked_absent:
            manual = "absent"  # deliberate seen-but-marked-absent review case
            first_seen_marked_absent = True
        else:
            manual = "present"
        marks.append({"student_id": row.student_id, "manual_status": manual})
    pd.DataFrame(marks).to_csv(destination, index=False)


def normalize_manual(frame, roster):
    if frame.empty:
        result = roster[["student_id", "name", "roll_no"]].copy()
        result["manual_attendance"] = "Not recorded"
        return result
    if "student_id" not in frame:
        return normalize_manual(pd.DataFrame(), roster)
    column = next((key for key in ("manual_attendance", "manual_status", "attendance", "present/absent", "present") if key in frame), None)
    if column is None:
        return normalize_manual(pd.DataFrame(), roster)
    values = frame[column].fillna("").astype(str).str.strip().str.lower()
    mapping = {"present": "Present", "yes": "Present", "1": "Present", "true": "Present",
               "absent": "Absent", "no": "Absent", "0": "Absent", "false": "Absent"}
    manual = pd.DataFrame({"student_id": frame.student_id.astype(str),
                           "manual_attendance": values.map(mapping).fillna("Not recorded")})
    manual = manual.drop_duplicates("student_id", keep="last")
    result = roster[["student_id", "name", "roll_no"]].merge(manual, on="student_id", how="left")
    result["manual_attendance"] = result.manual_attendance.fillna("Not recorded")
    return result


def classify(row):
    manual, video = row.manual_attendance, row.video_status
    if video == "Seen (low confidence)":
        return "Low confidence"
    if manual == "Present" and video == "Not seen":
        return "Possible proxy"
    if manual == "Absent" and video == "Seen":
        return "Seen but marked absent"
    if (manual == "Present" and video == "Seen") or (manual == "Absent" and video == "Not seen"):
        return "Match"
    return "Awaiting manual attendance"


def format_duration(seconds):
    seconds = max(0, int(round(float(seconds or 0))))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


def save_manual_mark(manual_path, roster, student_id, value):
    manual = normalize_manual(safe_csv(manual_path), roster)
    manual.loc[manual.student_id.astype(str) == str(student_id), "manual_attendance"] = value
    manual[["student_id", "manual_attendance"]].to_csv(manual_path, index=False)


def tone(value):
    text = str(value or "").lower()
    if any(word in text for word in ("proxy", "problem", "absent", "not seen")):
        return "danger"
    if any(word in text for word in ("low", "gap", "review", "pending", "flag")):
        return "warning"
    if any(word in text for word in ("match", "seen", "present", "confirmed", "high")):
        return "success"
    return "neutral"   


    


def render_reconciliation_table(frame):
    """Render the teacher table as accessible, theme-independent HTML."""
    rows = []
    for row in frame.itertuples(index=False):
        review_status = getattr(row, "review_status", "Pending")
        values = [
            f"<strong>{escape(str(row.name))}</strong><small>{escape(str(row.student_id))} · {escape(str(row.roll_no or '—'))}</small>",
            f"<span class='status-pill {tone(row.manual_attendance)}'>{escape(str(row.manual_attendance))}</span>",
            f"<span class='status-pill {tone(row.video_status)}'>{escape(str(row.video_status))}</span>",
            escape(str(row.visible_time)),
            f"<strong>{float(row.confidence_score):.0f}</strong><small>{escape(str(row.confidence))} confidence</small>",
            escape(str(row.flags if row.flags != "—" else "No flags")),
            f"<span class='status-pill {tone(review_status)}'>{escape(str(review_status))}</span>",
            f"<span class='status-pill {tone(row.reconciliation)}'>{escape(str(row.reconciliation))}</span>",
        ]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    headers = ("Student", "Teacher mark", "Video evidence", "Visible time", "Confidence", "Flags", "Review", "Reconciliation")
    header_html = "".join(f"<th>{escape(label)}</th>" for label in headers)
    st.markdown(
        "<div class='teacher-table-wrap'><table class='teacher-table'><thead><tr>" + header_html +
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_simple_table(frame):
    if frame.empty:
        st.caption("No records to display.")
        return
    headers = "".join(f"<th>{escape(str(column).replace('_', ' ').title())}</th>" for column in frame.columns)
    rows = []
    for values in frame.fillna("").astype(str).itertuples(index=False, name=None):
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>")
    st.markdown("<div class='teacher-table-wrap'><table class='teacher-table simple'><thead><tr>" +
        headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>", unsafe_allow_html=True)


def load_review_state(folder):
    value = read_json(Path(folder) / "review_state.json", {})
    return value if isinstance(value, dict) else {}


def save_review_state(folder, state):
    write_json(Path(folder) / "review_state.json", state)


def append_audit(folder, event):
    path = Path(folder) / "audit_log.csv"
    fields = ["timestamp", "teacher", "student_id", "old_value", "new_value", "reason"]
    with path.open("a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({field: event.get(field, "") for field in fields})


def evidence_path(folder, relative):
    if not isinstance(relative, str) or not relative:
        return None
    candidate = Path(folder) / relative
    if candidate.is_file():
        return candidate
    candidate = Path(folder) / "face_evidence" / relative
    return candidate if candidate.is_file() else None


def timeline_chart(timeline, report_row):
    intervals = timeline.get("visibility_intervals", []) if isinstance(timeline, dict) else []
    if not intervals:
        try:
            intervals = json.loads(report_row.get("visibility_intervals", "[]"))
        except (ValueError, TypeError):
            intervals = []
    parsed = []
    for item in intervals:
        start, end = (item.get("start_sec"), item.get("end_sec")) if isinstance(item, dict) else item[:2]
        if start is not None and end is not None and float(end) > float(start):
            parsed.append((float(start), float(end)))
    fig = go.Figure()
    for start, end in parsed:
        fig.add_trace(go.Bar(x=[end - start], y=["Visibility"], base=[start], orientation="h",
            marker_color="#248461", showlegend=False, hovertemplate=f"Visible evidence: {start:.1f}–{end:.1f}s<extra></extra>"))
    for index in range(len(parsed) - 1):
        start, end = parsed[index][1], parsed[index + 1][0]
        if end > start:
            fig.add_trace(go.Bar(x=[end - start], y=["Gap"], base=[start], orientation="h",
                marker_color="#E0E5EB", showlegend=False, hovertemplate=f"No visibility evidence: {start:.1f}–{end:.1f}s<extra></extra>"))
    if not parsed:
        fig.add_annotation(text="No visibility evidence recorded", showarrow=False)
    fig.update_layout(height=220, barmode="overlay", margin={"l": 18, "r": 18, "t": 18, "b": 46},
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"color": "#263448", "size": 14},
        xaxis={"title": "Seconds from video start", "gridcolor": "#E0E5EB", "zerolinecolor": "#E0E5EB"},
        yaxis={"showticklabels": False, "gridcolor": "#E0E5EB"})
    return fig


def styles():
    st.markdown("""
    <style>
    :root {--bg:#F3F5F7;--sidebar:#354154;--sidebar-secondary:#405069;--sidebar-active:#526178;--card:#FFFFFF;--surface:#FFFFFF;--ink:#263448;--muted:#667386;--line:#E0E5EB;--icon:#71849B;--success:#248461;--warning:#B87526;--danger:#B84757}
    .stApp {background:var(--bg);color:var(--ink)}
    [data-testid="stAppViewContainer"] {background:var(--bg)}
    .block-container {max-width:1500px;padding:1.15rem 2.2rem 3rem}
    [data-testid="stHeader"] {background:rgba(243,245,247,.96);backdrop-filter:blur(10px)}
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none!important}
    section[data-testid="stSidebar"] {display:block!important;background:var(--sidebar);border-right:1px solid #2E394B;min-width:260px}
    section[data-testid="stSidebar"] > div {background:var(--sidebar)}
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding:1.1rem .85rem}
    h1 {font-size:2.15rem!important;color:var(--ink);letter-spacing:-.035em;margin:.15rem 0}
    h2 {font-size:1.35rem!important;color:var(--ink);margin:1.15rem 0 .5rem}
    h3 {font-size:1.15rem!important;color:var(--ink)}
    p, label, [data-testid="stCaptionContainer"] {font-size:.94rem;color:var(--muted);line-height:1.5}
    div[data-testid="stHorizontalBlock"] {gap:1rem;align-items:stretch}
    div[data-testid="stVerticalBlock"] {gap:.8rem}
    [data-testid="stMetric"] {position:relative;background:var(--card);border:1px solid var(--line);border-left:3px solid var(--icon);border-radius:12px;padding:13px 16px;min-height:94px;box-shadow:0 3px 10px rgba(38,52,72,.045)}
    [data-testid="stMetricLabel"] {font-size:.82rem;font-weight:650;line-height:1.3;color:var(--muted);white-space:normal;overflow-wrap:anywhere;min-height:2.25em}
    [data-testid="stMetricValue"] {font-size:1.55rem;font-weight:720;line-height:1.15;color:var(--ink);overflow-wrap:anywhere}
    div.st-key-kpi_roster [data-testid="stMetric"],div.st-key-kpi_unknown_tracks [data-testid="stMetric"] {border-left-color:#71849B}
    div.st-key-kpi_seen [data-testid="stMetric"] {border-left-color:#248461}
    div.st-key-kpi_not_seen [data-testid="stMetric"] {border-left-color:#B84757}
    div.st-key-kpi_low_confidence [data-testid="stMetric"],div.st-key-kpi_flags [data-testid="stMetric"] {border-left-color:#B87526}
    [data-baseweb="select"] > div {background:var(--surface)!important;color:var(--ink)!important;border-color:var(--line)!important;border-radius:9px;min-height:42px}
    [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {background:var(--surface)!important;color:var(--ink)!important;border-color:var(--line)!important}
    [role="option"] {background:var(--surface)!important;color:var(--ink)!important}
    [role="option"]:hover {background:#EEF1F4!important}
    [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {background:var(--surface)!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:9px;min-height:42px}
    [data-testid="stTextInput"] input::placeholder,[data-testid="stTextArea"] textarea::placeholder {color:#7C8796!important}
    [data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus {border-color:#526178!important;box-shadow:0 0 0 3px rgba(82,97,120,.16)!important}
    [data-testid="stCheckbox"] label {font-size:.94rem!important;color:var(--muted)!important}
    [data-testid="stForm"] {background:var(--card);border:1px solid var(--line);border-radius:13px;padding:20px;box-shadow:0 3px 12px rgba(38,52,72,.045)}
    [data-testid="stAlert"] {margin:.4rem 0 .65rem;background:var(--card);border:1px solid var(--line);border-left:3px solid var(--icon);border-radius:10px;padding:11px 15px;box-shadow:0 2px 8px rgba(38,52,72,.04)}
    [data-testid="stAlert"][kind="warning"] {border-left-color:var(--warning)}
    [data-testid="stAlert"][kind="error"] {border-left-color:var(--danger)}
    [data-testid="stAlert"][kind="success"] {border-left-color:var(--success)}
    [data-testid="stAlert"] p,[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color:var(--ink)}
    .session-header {height:100%;box-sizing:border-box;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px 17px;margin:.15rem 0 .55rem;min-height:88px;box-shadow:0 3px 10px rgba(38,52,72,.04)}
    .eyebrow {font-size:.73rem;font-weight:750;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
    .sidebar-brand {background:var(--sidebar-secondary);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:15px 14px;margin-bottom:20px}
    .sidebar-brand-mark {display:grid;place-items:center;width:38px;height:38px;margin-bottom:13px;border-radius:10px;background:var(--sidebar-active);color:#fff;font-weight:800;font-size:18px}
    section[data-testid="stSidebar"] .sidebar-brand .brand-kicker {color:#F5F7FB;font-size:10px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}
    .sidebar-brand .brand-title {color:#fff;font-size:17px;line-height:1.25;font-weight:750;letter-spacing:-.02em;margin-top:6px}
    section[data-testid="stSidebar"] .sidebar-brand .brand-subtitle {color:#E0E7F0;font-size:11px;line-height:1.5;margin-top:8px}
    section[data-testid="stSidebar"] [data-testid="stRadio"] > [data-testid="stWidgetLabel"] {font-size:10px;font-weight:750;letter-spacing:.12em;color:#FFFFFF!important;text-transform:uppercase;margin:0 0 8px}
    section[data-testid="stSidebar"] [data-testid="stRadio"] > [data-testid="stWidgetLabel"] p {color:#FFFFFF!important;letter-spacing:.12em}
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {gap:5px}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] {box-sizing:border-box;width:100%;padding:9px 10px;margin:0;border:1px solid transparent;border-radius:9px;color:#FFFFFF!important;font-weight:700!important;transition:background .15s ease,border-color .15s ease}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] p,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] span,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] svg {color:#FFFFFF!important;fill:#FFFFFF;stroke:#FFFFFF;font-weight:700}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] p::first-letter {color:#FFFFFF!important}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {background:var(--sidebar-secondary);border-color:rgba(255,255,255,.12);color:#FFFFFF!important}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover p,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover span,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover svg {color:#FFFFFF!important;fill:#FFFFFF;stroke:#FFFFFF}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover p::first-letter {color:#FFFFFF!important}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {background:var(--sidebar-active);border-color:rgba(255,255,255,.2);color:#FFFFFF;font-weight:700}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) span,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) svg {color:#FFFFFF!important;fill:#E0E7F0;stroke:#E0E7F0}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p::first-letter {color:#FFFFFF!important}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:focus-within {outline:2px solid #E0E7F0;outline-offset:2px}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:disabled) {color:#AAB5C4!important;font-weight:500!important;cursor:not-allowed}
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:disabled) p,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:disabled) span,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:disabled) svg {color:#AAB5C4!important;fill:#AAB5C4;stroke:#AAB5C4}
    section[data-testid="stSidebar"] [data-testid="stRadio"] input {accent-color:#fff}
    section[data-testid="stSidebar"] .sidebar-footer {border-top:1px solid rgba(255,255,255,.16);margin-top:22px;padding-top:12px;color:#FFFFFF;font-size:11px;line-height:1.55}
    .section-card {background:var(--card);border:1px solid var(--line);border-radius:13px;padding:19px 21px;margin:11px 0;box-shadow:0 3px 12px rgba(38,52,72,.04)}
    .teacher-table-wrap {overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 12px rgba(38,52,72,.045);margin:11px 0 18px}
    .teacher-table {width:100%;border-collapse:collapse;background:transparent;color:#263448;font-size:13px;min-width:940px}
    .teacher-table th {background:#F3F5F7;color:#526178;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.075em;font-weight:750;padding:13px 14px;border-bottom:1px solid #E0E5EB;white-space:nowrap}
    .teacher-table td {padding:12px 14px;border-bottom:1px solid #E8EBEF;vertical-align:middle;line-height:1.45}
    .teacher-table tbody tr:nth-child(even) {background:#FAFBFC}
    .teacher-table tbody tr:last-child td {border-bottom:0}
    .teacher-table tbody tr:hover {background:#F1F4F7}
    .teacher-table small {display:block;color:#667386;font-size:11px;margin-top:3px}
    .teacher-table .status-pill {display:inline-block;border:1px solid transparent;border-radius:7px;padding:4px 8px;font-size:11px;font-weight:700;white-space:nowrap}
    .status-pill.success {background:#E7F3EE;color:#1E684D;border-color:#C8E4D7}
    .status-pill.warning {background:#FBF2E6;color:#885519;border-color:#EFDBBD}
    .status-pill.danger {background:#F9EAEC;color:#963947;border-color:#EFD1D5}
    .status-pill.neutral {background:#EEF1F4;color:#526178;border-color:#DDE3E9}
    button[kind="primary"] {background:#405069!important;border:1px solid #405069!important;color:#fff!important;min-height:44px;padding:0 18px;font-size:.94rem;font-weight:680;border-radius:9px;box-shadow:0 2px 6px rgba(38,52,72,.12);transition:background .16s ease,box-shadow .16s ease}
    button[kind="primary"]:hover {background:#354154!important;border-color:#354154!important;box-shadow:0 4px 10px rgba(38,52,72,.16)}
    button[kind="primary"]:focus-visible,button[kind="secondary"]:focus-visible {outline:3px solid #71849B!important;outline-offset:2px}
    button[kind="secondary"] {background:#fff!important;color:#354154!important;min-height:44px;padding:0 18px;font-size:.94rem;font-weight:620;border-radius:9px;border-color:#C9D1DB!important;transition:background .16s ease,border-color .16s ease}
    button[kind="secondary"]:hover {background:#F3F5F7!important;border-color:#71849B!important}
    [class*="st-key-Mark_Present_"] button {background:#248461!important;border-color:#248461!important;color:#fff!important}
    [class*="st-key-Mark_Present_"] button:hover {background:#1D6F51!important;border-color:#1D6F51!important}
    [class*="st-key-Mark_Absent_"] button {background:#B84757!important;border-color:#B84757!important;color:#fff!important}
    [class*="st-key-Mark_Absent_"] button:hover {background:#963947!important;border-color:#963947!important}
    .page-heading {display:flex;align-items:center;justify-content:space-between;gap:16px;margin:0 0 12px;padding:0 0 12px;border-bottom:1px solid #E0E5EB}
    .page-heading .page-eyebrow {color:#667386;font-size:10px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}
    .page-heading .page-title {color:#263448;font-size:23px;font-weight:750;letter-spacing:-.025em;margin-top:2px}
    .page-heading .page-context {color:#667386;font-size:12px;text-align:right}
    [data-testid="stPlotlyChart"] {background:#FFFFFF;border:1px solid #E0E5EB;border-radius:12px;padding:8px 10px;box-shadow:0 3px 10px rgba(38,52,72,.045)}
    [data-testid="stImage"] img {border:1px solid #E0E5EB;border-radius:10px}
    @media (max-width: 950px) {
      .block-container {padding:1rem 1.15rem 2rem}
      section[data-testid="stSidebar"] {min-width:225px}
    }
    @media (max-width: 700px) {
      section[data-testid="stSidebar"] {min-width:0}
      div[data-testid="stHorizontalBlock"] {flex-wrap:wrap}
      div[data-testid="stHorizontalBlock"] > [data-testid="column"] {min-width:min(100%,210px)!important}
      [data-testid="stMetric"] {min-height:88px}
      .page-heading .page-title {font-size:20px}
    }
    </style>
    """, unsafe_allow_html=True)


def format_session(session):
    date_text = session.get("created_at", "")[:16].replace("T", " ")
    return f"{session.get('class_id', 'Class')} · {session.get('video_name', 'Video')} · {date_text}"


def main():
    st.set_page_config(page_title="AI Attendance Anomaly Detector", page_icon="▦", layout="wide")
    styles()
    navigation = ["Overview", "Attendance review", "Student details", "Audit log & export", "Export"]
    navigation_icons = {
        "Overview": "▦",
        "Attendance review": "✓",
        "Student details": "◎",
        "Audit log & export": "↗",
        "Export": "⇩",
    }
    navigation_labels = {
        "Overview": "Overview",
        "Attendance review": "Attendance Review",
        "Student details": "Student Details",
        "Audit log & export": "Audit Log",
        "Export": "Export",
    }
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="sidebar-brand-mark">◉</div>
          <div class="brand-kicker">Attendance workspace</div>
          <div class="brand-title">AI Attendance<br>Anomaly Detector</div>
          <div class="brand-subtitle">Teacher review · Video visibility evidence</div>
        </div>
        """, unsafe_allow_html=True)
        active_section = st.radio(
            "Workspace",
            navigation,
            format_func=lambda label: f"{navigation_icons[label]}  {navigation_labels[label]}",
            label_visibility="visible",
            key="dashboard_navigation",
        )
        st.markdown(
            '<div class="sidebar-footer">Manual attendance remains the record.<br>Review decisions are recorded in the audit log.</div>',
            unsafe_allow_html=True,
        )

    videos = discover_videos()
    top_left, top_right = st.columns([1.5, 1])
    with top_left:
        st.subheader("Class Session")
        class_id = st.text_input("Class / section", value="CSE-A", label_visibility="collapsed", placeholder="Class / section")
        demo_mode = st.checkbox("Temporary demo identities from this video (not an accuracy evaluation)")
        video_names = [path.name for path in videos]
        if video_names:
            selected_video_name = st.selectbox("Classroom video", video_names, label_visibility="collapsed")
        else:
            selected_video_name = None
            st.info("No classroom videos are available in the configured local video library. Ask the administrator to add a session video.")
    with top_right:
        teacher = st.text_input("Teacher", value="Teacher")
        start_button = st.button("Start class session", type="primary", disabled=not bool(videos), use_container_width=True)

    if start_button and selected_video_name:
        if not demo_mode and not DATABASE_PATH.is_file():
            st.error("The enrolment database is unavailable. An administrator must set ATTENDANCE_DB_PATH to the M1 SQLite database, then restart the dashboard.")
            st.stop()
        selected_path = next(path for path in videos if path.name == selected_video_name)
        with st.status("Starting class session…", expanded=True) as status:
            st.write("Processing classroom video and identifying visible students…")
            live_progress = st.empty()
            def show_progress(label, line):
                if line:
                    live_progress.caption(f"{label} · {line[:180]}")
            session, success = run_backend(selected_path, class_id, teacher.strip() or "Teacher",
                demo=demo_mode, progress_callback=show_progress)
            if success:
                status.update(label="Session processing complete", state="complete", expanded=False)
                st.session_state["active_session_id"] = session["session_id"]
            else:
                status.update(label="Session processing failed", state="error", expanded=True)
                st.error("The session could not be processed. Ask the administrator to review the local processing log.")
        if success:
            st.rerun()

    sessions = load_sessions(demo_mode)
    if not sessions:
        if demo_mode:
            st.warning("DEMO: temporary identities are derived from this same video. This is not an accuracy evaluation.")
        st.markdown('<div class="session-header">Choose a class and classroom video, then start the session. Processing and report generation happen automatically.</div>', unsafe_allow_html=True)
        st.stop()
    session_ids = [item["session_id"] for item in sessions]
    active_id = st.session_state.get("active_session_id", session_ids[0])
    if active_id not in session_ids:
        active_id = session_ids[0]
    selected_id = st.selectbox("Session", session_ids, index=session_ids.index(active_id),
        format_func=lambda sid: format_session(next(item for item in sessions if item["session_id"] == sid)),
        label_visibility="collapsed")
    session = next(item for item in sessions if item["session_id"] == selected_id)

    header_cols = st.columns([1.2, 1.7, 1])
    with header_cols[0]:
        st.markdown(f"<div class='session-header'><div class='eyebrow'>Class / section</div><h3>{session['class_id']}</h3></div>", unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown(f"<div class='session-header'><div class='eyebrow'>Video</div><h3>{session['video_name']}</h3></div>", unsafe_allow_html=True)
    with header_cols[2]:
        status_color = "#248461" if session.get("status") == "Completed" else "#B87526" if session.get("status") == "Running" else "#B84757"
        st.markdown(f"<div class='session-header'><div class='eyebrow'>Processing status</div><h3 style='color:{status_color}'>{session.get('status','Unknown')}</h3></div>", unsafe_allow_html=True)
    if session.get("status") != "Completed":
        if session.get("status") == "Failed":
            st.error("Processing did not complete. The local backend log contains technical details for the administrator.")
        st.stop()
    session_database = session_folder(session, "database_path") if session.get("database_path") else DATABASE_PATH
    if session.get("demo_mode"):
        st.warning("DEMO: same-video enrollment; identity labels are temporary examples, not validated recognition or accuracy results.")
        if session.get("sample_manual_scenario"):
            st.info("Sample manual attendance is prefilled to demonstrate reconciliation categories. Edit it before reviewing; it is not real roll call.")

    report_dir = session_folder(session, "report_dir")
    evidence_dir = session_folder(session, "processing_dir")
    report, report_error = load_report(report_dir / "report.csv")
    if report_error:
        st.error(report_error)
        st.stop()
    report["student_id"] = report.student_id.astype(str)
    roster = load_roster(report, session_database)
    merged = roster.merge(report, on=["student_id", "name"], how="left", suffixes=("", "_report"))
    if "roll_no" not in merged:
        merged["roll_no"] = ""
    merged["video_status"] = merged.status.fillna("Not seen")
    for column, default in (("visible_percent", 0.0), ("confidence_score", 0.0),
                            ("total_visible_seconds", 0.0), ("mean_recognition_similarity", 0.0),
                            ("face_detected_visible_percent", 0.0),
                            ("mean_person_detection_confidence", 0.0), ("face_samples", 0)):
        if column not in merged:
            merged[column] = default
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(default)
    for column, default in (("confidence", "Low"), ("flags", "—"), ("first_seen", ""), ("last_seen", ""), ("visibility_intervals", "[]")):
        if column not in merged:
            merged[column] = default
        merged[column] = merged[column].fillna(default)
    merged["visible_time"] = merged.total_visible_seconds.map(format_duration)
    track_table = safe_csv(evidence_dir / "identities.csv")
    track_id_count = int(track_table.track_id.nunique()) if "track_id" in track_table else 0
    unknown_track_count = int(track_table.student_id.isna().sum()) if "student_id" in track_table else 0

    report_dir.mkdir(parents=True, exist_ok=True)
    manual_path = report_dir / "manual_attendance.csv"
    manual = normalize_manual(safe_csv(manual_path), roster)
    st.markdown(
        f"<div class='page-heading'><div><div class='page-eyebrow'>Class attendance workspace</div>"
        f"<div class='page-title'>{escape(navigation_labels[active_section])}</div></div>"
        f"<div class='page-context'>{escape(format_session(session))}</div></div>",
        unsafe_allow_html=True,
    )

    seen_count = int(merged.video_status.isin(["Seen", "Seen (low confidence)"]).sum())
    not_seen_count = int(merged.video_status.eq("Not seen").sum())
    low_count = int(merged.video_status.eq("Seen (low confidence)").sum())
    flagged_count = int(merged["flags"].astype(str).ne("—").sum())
    export_state = load_review_state(report_dir)
    export = merged.merge(manual[["student_id", "manual_attendance"]], on="student_id", how="left")
    export["manual_attendance"] = export.manual_attendance.fillna("Not recorded")
    export["reconciliation"] = export.apply(classify, axis=1)
    export["review_status"] = export.student_id.map(
        lambda sid: export_state.get(str(sid), {}).get("review_status", "Pending"))
    export["final_attendance"] = export.student_id.map(
        lambda sid: export_state.get(str(sid), {}).get("final_attendance", ""))
    export["review_reason_or_note"] = export.student_id.map(
        lambda sid: export_state.get(str(sid), {}).get("reason", export_state.get(str(sid), {}).get("note", "")))
    final_columns = ["student_id", "name", "roll_no", "manual_attendance", "video_status", "reconciliation",
        "final_attendance", "review_status", "review_reason_or_note", "visible_percent", "confidence", "flags"]
    final = export[[column for column in final_columns if column in export]]
    final.to_csv(report_dir / "final_attendance.csv", index=False)

    if active_section == 'Overview':
        st.subheader("Session overview")
        metric_labels = ["Roster", "Seen", "Not seen", "Low confidence", "Unknown tracks", "Flags"]
        metric_values = [len(merged), seen_count, not_seen_count, low_count, unknown_track_count, flagged_count]
        metric_keys = ["kpi_roster", "kpi_seen", "kpi_not_seen", "kpi_low_confidence", "kpi_unknown_tracks", "kpi_flags"]
        for start in (0, 3):
            cols = st.columns(3)
            for col, key, label, value in zip(cols, metric_keys[start:start + 3],
                                               metric_labels[start:start + 3], metric_values[start:start + 3]):
                with col.container(key=key):
                    st.metric(label, value)
        st.caption(f"YOLO tracker produced {track_id_count} track IDs. Track IDs can split or merge and are not a headcount.")
        st.info("Not visible does not mean a student left the room. The teacher makes every attendance decision.")
        st.subheader("Reconciliation summary")
        preview = merged.merge(manual[["student_id", "manual_attendance"]], on="student_id", how="left")
        preview["manual_attendance"] = preview.manual_attendance.fillna("Not recorded")
        preview["reconciliation"] = preview.apply(classify, axis=1)
        preview["visible_time"] = preview.apply(
            lambda item: f"{item.visible_time} · {item.visible_percent:.1f}%", axis=1)
        render_reconciliation_table(preview[["student_id", "name", "roll_no", "manual_attendance", "video_status",
            "visible_time", "confidence_score", "confidence", "flags", "reconciliation"]])

    if active_section == 'Attendance review':
        st.subheader("Attendance reconciliation")
        st.caption("Select a student and record the teacher's mark. Video evidence never sets attendance automatically.")
        quick_student = st.selectbox("Student to mark", merged.student_id.tolist(),
            format_func=lambda sid: f"{merged.loc[merged.student_id == sid, 'name'].iloc[0]} · {sid}",
            key=f"quick_mark_{session['session_id']}")
        current_manual = manual.loc[manual.student_id.astype(str) == str(quick_student), "manual_attendance"]
        old_manual = current_manual.iloc[0] if not current_manual.empty else "Not recorded"
        st.caption(f"Current teacher mark: **{old_manual}**")
        mark_cols = st.columns([1, 1, 1, 4])
        for column, value, label in ((mark_cols[0], "Present", "Mark Present"),
                                     (mark_cols[1], "Absent", "Mark Absent"),
                                     (mark_cols[2], "Not recorded", "Clear mark")):
            if column.button(label, key=f"{label}_{session['session_id']}_{quick_student}",
                              type="primary" if value != "Not recorded" else "secondary", use_container_width=True):
                save_manual_mark(manual_path, roster, quick_student, value)
                append_audit(report_dir, {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "teacher": teacher, "student_id": quick_student, "old_value": old_manual,
                    "new_value": value, "reason": "Teacher updated manual attendance mark"})
                st.rerun()
        review = merged.merge(manual[["student_id", "manual_attendance"]], on="student_id", how="left")
        review["manual_attendance"] = review.manual_attendance.fillna("Not recorded")
        review["reconciliation"] = review.apply(classify, axis=1)
        state = load_review_state(report_dir)
        review["review_status"] = review.student_id.map(lambda sid: state.get(str(sid), {}).get("review_status", "Pending"))
        if st.button("Confirm all matches", help="Confirms only rows categorized Match with a recorded manual value."):
            matches = review[(review.reconciliation == "Match") & review.manual_attendance.isin(["Present", "Absent"])]
            for row in matches.itertuples(index=False):
                old = state.get(str(row.student_id), {}).get("final_attendance", "Pending")
                state[str(row.student_id)] = {"review_status": "Confirmed", "final_attendance": row.manual_attendance,
                    "reason": "Teacher confirmed match", "note": ""}
                append_audit(report_dir, {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "teacher": teacher, "student_id": row.student_id, "old_value": old,
                    "new_value": row.manual_attendance, "reason": "Teacher confirmed match"})
            save_review_state(report_dir, state)
            st.rerun()
        review["visible_time"] = review.apply(
            lambda item: f"{item.visible_time} · {item.visible_percent:.1f}%", axis=1)
        render_reconciliation_table(review[["student_id", "name", "roll_no", "manual_attendance", "video_status",
            "visible_time", "confidence_score", "confidence", "flags", "reconciliation", "review_status"]])

    if active_section == 'Student details':
        st.subheader("Student details")
        chosen = st.selectbox("Select student", merged.student_id.tolist(),
            format_func=lambda sid: f"{merged.loc[merged.student_id == sid, 'name'].iloc[0]} · {sid}")
        row = merged[merged.student_id == chosen].iloc[0]
        timeline_data = read_json(report_dir / "timelines.json", {})
        timeline = timeline_data.get(str(chosen), {}) if isinstance(timeline_data, dict) else {}
        st.markdown(f"### {row['name']} · {chosen} · Roll {row['roll_no'] or '—'}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total visible time", format_duration(row.total_visible_seconds))
        d2.metric("Visible share", f"{row.visible_percent:.1f}%")
        d3.metric("Confidence", f"{row.confidence_score:.1f} / 100 · {row.confidence}")
        d4.metric("Face samples", int(row.face_samples))
        detail_left, detail_right = st.columns([1.7, 1])
        with detail_left:
            st.plotly_chart(timeline_chart(timeline, row), use_container_width=True, config={"displayModeBar": False})
            st.markdown("**Flags**")
            if "Early last seen" in str(row["flags"]):
                st.warning("Visibility ended well before the video ended. This needs teacher review and does not confirm the student left.")
            other_flags = [flag.strip() for flag in str(row["flags"]).split(",")
                           if flag.strip() not in {"", "—", "Early last seen"}]
            st.write(", ".join(other_flags) if other_flags else ("No other flags" if row["flags"] != "—" else "No flags"))
            st.caption(f"Identity similarity: {row.mean_recognition_similarity:.3f} · Person detector confidence: "
                       f"{row.mean_person_detection_confidence:.3f} · Face-detected visible time: "
                       f"{row.face_detected_visible_percent:.1f}% · Face samples: {int(row.face_samples)}")
        with detail_right:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(row.confidence_score),
                number={"suffix": "/100", "font": {"color": "#263448", "size": 30}},
                title={"text": f"Confidence · {row.confidence}", "font": {"size": 16, "color": "#667386"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#667386", "tickfont": {"color": "#667386"}},
                       "bar": {"color": "#526178"},
                       "bgcolor": "#FFFFFF", "bordercolor": "#E0E5EB",
                       "steps": [{"range": [0, 100], "color": "#EEF1F4"}]}))
            gauge.update_layout(height=220, margin={"l": 20, "r": 20, "t": 35, "b": 10},
                                paper_bgcolor="#FFFFFF")
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown("#### Face evidence")
        identities = safe_csv(evidence_dir / "identities.csv")
        if not identities.empty and {"student_id", "best_face_file"}.issubset(identities.columns):
            evidence_rows = identities[identities.student_id.astype(str) == str(chosen)].head(4)
            evidence_cols = st.columns(max(1, min(4, len(evidence_rows))))
            shown = 0
            for evidence in evidence_rows.itertuples(index=False):
                image_path = evidence_path(evidence_dir, getattr(evidence, "best_face_file", ""))
                if image_path:
                    evidence_cols[shown % len(evidence_cols)].image(str(image_path),
                        caption=f"{getattr(evidence, 'best_face_time_sec', '')} sec", width=220)
                    shown += 1
            if not shown:
                st.caption("No face evidence crop is available for this student.")
        else:
            st.caption("No face evidence is available for this session.")

        state = load_review_state(report_dir)
        current = state.get(str(chosen), {})
        teacher_manual = safe_csv(manual_path)
        manual_row = teacher_manual[teacher_manual.student_id.astype(str) == str(chosen)] if not teacher_manual.empty else pd.DataFrame()
        manual_value = manual_row.iloc[0].get("manual_attendance", "Not recorded") if not manual_row.empty else "Not recorded"
        st.markdown("#### Teacher review actions")
        left, right = st.columns(2)
        with left:
            with st.form(f"confirm_{session['session_id']}_{chosen}"):
                st.write(f"Manual value: **{manual_value}**")
                confirm = st.form_submit_button("Confirm", disabled=manual_value not in ("Present", "Absent"))
                if confirm:
                    old = current.get("final_attendance", "Pending")
                    state[str(chosen)] = {"review_status": "Confirmed", "final_attendance": manual_value,
                                          "reason": "Teacher confirmed manual record", "note": ""}
                    append_audit(report_dir, {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "teacher": teacher, "student_id": chosen, "old_value": old, "new_value": manual_value,
                        "reason": "Teacher confirmed manual record"})
                    save_review_state(report_dir, state)
                    st.rerun()
            with st.form(f"correct_{session['session_id']}_{chosen}"):
                corrected_value = st.selectbox("Correct final attendance", ["Present", "Absent"], key=f"correct_value_{chosen}")
                correction_reason = st.text_input("Reason (required)", key=f"correct_reason_{chosen}")
                correct = st.form_submit_button("Save correction")
                if correct:
                    if not correction_reason.strip():
                        st.error("Provide a reason for the correction.")
                    else:
                        old = current.get("final_attendance", manual_value)
                        state[str(chosen)] = {"review_status": "Corrected", "final_attendance": corrected_value,
                                              "reason": correction_reason.strip(), "note": ""}
                        append_audit(report_dir, {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "teacher": teacher, "student_id": chosen, "old_value": old,
                            "new_value": corrected_value, "reason": correction_reason.strip()})
                        save_review_state(report_dir, state)
                        st.rerun()
        with right:
            with st.form(f"followup_{session['session_id']}_{chosen}"):
                note = st.text_area("Follow-up note")
                followup = st.form_submit_button("Save follow-up")
                if followup:
                    if not note.strip():
                        st.error("Enter a follow-up note.")
                    else:
                        old = current.get("review_status", "Pending")
                        state[str(chosen)] = {**current, "review_status": "Follow-up", "note": note.strip()}
                        append_audit(report_dir, {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "teacher": teacher, "student_id": chosen, "old_value": old,
                            "new_value": "Follow-up", "reason": note.strip()})
                        save_review_state(report_dir, state)
                        st.rerun()

        st.markdown("#### Unknown faces")
        unknown = safe_csv(evidence_dir / "unknown_tracks.csv")
        if unknown.empty or "track_id" not in unknown:
            st.caption("There are no unknown face tracks in this session.")
        else:
            track_ids = unknown.track_id.astype(str).tolist()
            track_id = st.selectbox("Unknown track to review", track_ids)
            unknown_row = unknown[unknown.track_id.astype(str) == track_id].iloc[0]
            unknown_path = evidence_path(evidence_dir, unknown_row.get("best_face_file", ""))
            if unknown_path:
                st.image(str(unknown_path), caption="Unidentified face evidence", width=220)
            with st.form(f"unknown_assign_{session['session_id']}_{track_id}"):
                roster_ids = roster.student_id.astype(str).tolist()
                assigned_student = st.selectbox("Assign this track to a student", roster_ids,
                    format_func=lambda sid: f"{roster.loc[roster.student_id == sid, 'name'].iloc[0]} · {sid}")
                assign = st.form_submit_button("Save teacher assignment")
                if assign:
                    assignments_path = report_dir / "unknown_assignments.csv"
                    assignments = safe_csv(assignments_path)
                    if assignments.empty:
                        assignments = pd.DataFrame(columns=["track_id", "assigned_student_id", "teacher", "timestamp"])
                    assignments = assignments[assignments.track_id.astype(str) != track_id]
                    assignments.loc[len(assignments)] = [track_id, assigned_student, teacher,
                        datetime.now().astimezone().isoformat(timespec="seconds")]
                    assignments.to_csv(assignments_path, index=False)
                    append_audit(report_dir, {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "teacher": teacher, "student_id": assigned_student, "old_value": f"Unknown track {track_id}",
                        "new_value": assigned_student, "reason": "Teacher assigned unknown face track"})
                    st.success("Assignment saved for teacher review. It does not change enrolment or attendance.")
                    st.rerun()

    if active_section == 'Audit log & export':
        st.subheader("Audit log")
        audit = safe_csv(report_dir / "audit_log.csv")
        if audit.empty:
            st.caption("No teacher review actions have been recorded for this session.")
        else:
            render_simple_table(audit.sort_values("timestamp", ascending=False))
    if active_section == "Export":
        st.subheader("Final attendance export")
        st.caption("Pending rows have a blank final value. Only teacher-confirmed or corrected values are exported as final attendance.")
        st.download_button("Download final attendance", final.to_csv(index=False), "final_attendance.csv",
                           "text/csv", type="primary")


if __name__ == "__main__":
    main()
