import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Construction Safety Monitor",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Construction Site Safety Monitoring")
st.caption("AI-assisted video analytics prototype | YOLO detection + ByteTrack tracking + rule-based zone monitoring")

st.info(
    "This dashboard summarizes exported notebook results. "
    "It is a prototype for one processed video, not a live CCTV system. "
    "Risk levels are rule-based and are not validated accident probabilities."
)

st.sidebar.header("Load project outputs")
st.sidebar.write("Upload the CSV files exported from the Colab notebook.")

uploaded_tracking = st.sidebar.file_uploader("worker_tracking_data.csv", type="csv")
uploaded_movement = st.sidebar.file_uploader("worker_movement_summary.csv", type="csv")
uploaded_zone = st.sidebar.file_uploader("restricted_zone_observations.csv", type="csv")
uploaded_events = st.sidebar.file_uploader("risk_event_report.csv", type="csv")

def read_csv_upload(uploaded, filename):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    local_path = Path("construction_safety_outputs") / filename
    if local_path.exists():
        return pd.read_csv(local_path)
    return None

person_df = read_csv_upload(uploaded_tracking, "worker_tracking_data.csv")
movement_df = read_csv_upload(uploaded_movement, "worker_movement_summary.csv")
zone_df = read_csv_upload(uploaded_zone, "restricted_zone_observations.csv")
events_df = read_csv_upload(uploaded_events, "risk_event_report.csv")

if person_df is None or zone_df is None or events_df is None:
    st.warning(
        "Upload the required CSV files in the sidebar. "
        "At minimum, upload worker_tracking_data.csv, "
        "restricted_zone_observations.csv, and risk_event_report.csv."
    )
    st.markdown("""
    **Where do I get these files?**
    1. Run the export cells at the end of the Colab notebook.
    2. Download `construction_safety_project_results.zip`.
    3. Extract it on your computer.
    4. Upload the CSV files using the sidebar.
    """)
    st.stop()

required_person = {"track_id", "frame", "timestamp", "bottom_x", "bottom_y"}
required_zone = {"track_id"}
required_events = {"track_id", "event_id", "risk_level", "risk_score", "duration_seconds"}

missing_person = required_person - set(person_df.columns)
missing_zone = required_zone - set(zone_df.columns)
missing_events = required_events - set(events_df.columns)

if missing_person or missing_zone or missing_events:
    st.error(
        f"CSV schema mismatch. Missing tracking columns: {sorted(missing_person)}; "
        f"zone columns: {sorted(missing_zone)}; event columns: {sorted(missing_events)}."
    )
    st.stop()

# Normalize numeric fields where possible
for col in ["track_id", "frame", "timestamp", "bottom_x", "bottom_y"]:
    person_df[col] = pd.to_numeric(person_df[col], errors="coerce")
zone_df["track_id"] = pd.to_numeric(zone_df["track_id"], errors="coerce")
for col in ["track_id", "event_id", "risk_score", "duration_seconds"]:
    if col in events_df.columns:
        events_df[col] = pd.to_numeric(events_df[col], errors="coerce")

total_tracks = int(person_df["track_id"].nunique())
total_observations = int(len(person_df))
zone_observations = int(len(zone_df))
event_count = int(len(events_df))
high_events = int((events_df["risk_level"].astype(str).str.lower() == "high").sum())

st.subheader("Safety monitoring overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tracked workers", f"{total_tracks:,}")
c2.metric("Person observations", f"{total_observations:,}")
c3.metric("Zone observations", f"{zone_observations:,}")
c4.metric("Intrusion episodes", f"{event_count:,}")
c5.metric("High-rule-score events", f"{high_events:,}")

left, right = st.columns(2)
with left:
    st.subheader("Risk event classification")
    risk_order = ["Low", "Medium", "High"]
    risk_counts = (
        events_df["risk_level"].astype(str).str.title()
        .value_counts().reindex(risk_order, fill_value=0)
        .rename_axis("Risk level").reset_index(name="Event count")
    )
    fig_risk = px.bar(
        risk_counts, x="Risk level", y="Event count",
        color="Risk level", category_orders={"Risk level": risk_order},
        title="Number of events by rule-based risk level"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with right:
    st.subheader("Zone observations by track")
    zone_counts = (
        zone_df.groupby("track_id").size()
        .reset_index(name="Observations")
    )
    fig_zone = px.bar(
        zone_counts, x="track_id", y="Observations",
        title="Observations inside the configured sample zone",
        labels={"track_id": "Track ID"}
    )
    st.plotly_chart(fig_zone, use_container_width=True)

st.subheader("Worker movement trajectories")
if {"bottom_x", "bottom_y", "track_id"}.issubset(person_df.columns):
    trajectory = person_df.dropna(subset=["bottom_x", "bottom_y", "track_id"]).copy()
    trajectory["track_id"] = trajectory["track_id"].astype(int).astype(str)
    fig_traj = px.line(
        trajectory.sort_values(["track_id", "frame"]),
        x="bottom_x", y="bottom_y", color="track_id",
        hover_data=[c for c in ["frame", "timestamp", "confidence"] if c in trajectory.columns],
        title="Tracked worker paths in image coordinates",
        labels={"bottom_x": "X coordinate (pixels)", "bottom_y": "Y coordinate (pixels)", "track_id": "Track ID"}
    )
    fig_traj.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_traj, use_container_width=True)
else:
    st.warning("Trajectory columns are unavailable in the tracking CSV.")

st.subheader("Risk event report")
display_events = events_df.copy()
preferred = [c for c in [
    "track_id", "event_id", "entry_frame", "exit_frame",
    "entry_time", "exit_time", "observations",
    "avg_confidence", "duration_seconds", "risk_score", "risk_level"
] if c in display_events.columns]
st.dataframe(display_events[preferred], use_container_width=True, hide_index=True)

if movement_df is not None:
    st.subheader("Worker movement summary")
    st.dataframe(movement_df, use_container_width=True, hide_index=True)

with st.expander("Interpretation and limitations"):
    st.markdown("""
    - A **zone observation** is a frame-level worker footpoint inside the manually configured rectangle.
    - An **intrusion episode** groups consecutive in-zone observations for a track.
    - Risk scores and Low/Medium/High labels are generated by the notebook's duration-based rules.
    - These labels are not machine-learning predictions, accident probabilities, or independently validated safety thresholds.
    - Coordinates and movement values are in pixels, not metres or km/h.
    - Results represent the uploaded video only; this app does not connect to live CCTV.
    """)

st.caption("Construction Safety Monitoring Prototype | Built with Streamlit and Plotly")
