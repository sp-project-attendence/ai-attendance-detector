"""Step 8: teacher dashboard.   Run: streamlit run dashboard.py"""
import json, os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from analyze import Params, analyze

OUT = "output"
st.set_page_config(page_title="AI Attendance Anomaly Detector", layout="wide")
st.title("AI Attendance Anomaly Detector")

meta = json.load(open(f"{OUT}/meta.json"))
tracks = pd.read_csv(f"{OUT}/tracks.csv")

with st.sidebar:
    st.header("Detection rules")
    p = Params(
        gap_tol=st.slider("Occlusion tolerance (sec)", 2, 60, 15),
        min_track_sec=st.slider("Min. track length (sec)", 5, 120, 30),
        late_frac=st.slider("Late if arrives after (% of class)", 1, 50, 10) / 100,
        early_frac=st.slider("Early exit if leaves before last (% of class)", 1, 50, 10) / 100,
        exits_flag=st.slider("Frequent exits: absences >=", 1, 10, 3),
    )

df, intervals = analyze(tracks, meta["duration"], meta["step"], p)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students detected", len(df))
c2.metric("Class duration", f"{meta['duration'] / 60:.1f} min")
c3.metric("Avg attendance", f"{df['Attendance %'].mean():.0f}%")
c4.metric("Flagged", int((df["Anomaly"] != "-").sum()))

left, right = st.columns([1, 1])
with left:
    st.subheader("Processed video (detections + track IDs)")
    vid = f"{OUT}/annotated.mp4"
    if os.path.exists(vid):
        st.video(vid)
    else:
        st.info("Run the ffmpeg conversion step to show the annotated video here.")
with right:
    st.subheader("Presence timeline")
    fig = go.Figure()
    for name, ivs in intervals.items():
        for s, e in ivs:
            fig.add_trace(go.Bar(y=[name], x=[(e - s) / 60], base=[s / 60], orientation="h",
                                 marker_color="#2e7d32", showlegend=False,
                                 hovertemplate=f"{name}<br>{s/60:.1f}-{e/60:.1f} min<extra></extra>"))
    fig.update_layout(barmode="overlay", xaxis_title="Minutes into class",
                      yaxis=dict(autorange="reversed"), height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Attendance report")
color = {"Present": "#c8e6c9", "Partial": "#fff9c4", "Absent": "#ffcdd2"}
st.dataframe(df.style.map(lambda v: f"background-color: {color.get(v, '')}", subset=["Status"]),
             use_container_width=True, hide_index=True)
st.download_button("Download CSV", df.to_csv(index=False), "attendance.csv")
