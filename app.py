from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
from ultralytics import YOLO

from safety_pipeline import inspect_video, normalized_zone_to_pixels, process_video

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "app_data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Construction Safety Intelligence", page_icon="🏗️", layout="wide")
st.title("🏗️ Construction Site Safety Monitoring")
st.caption("Multi-video computer-vision prototype | YOLO11 + ByteTrack + worker trajectories + rule-based zone monitoring")
st.info(
    "This application processes uploaded recorded videos sequentially. Each video receives an independent "
    "video_id and tracker run. Risk scores are transparent duration-based rules, not validated accident probabilities."
)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(weights: str):
    return YOLO(weights)


def persist_upload(uploaded_file) -> Path:
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()[:12]
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    target = UPLOAD_DIR / f"{digest}_{safe_name}"
    if not target.exists():
        target.write_bytes(content)
    return target


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    summaries = []
    tracking, movement, zones, events = [], [], [], []
    for folder in sorted(RESULTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        summary_path = folder / "video_summary.json"
        if not summary_path.exists():
            continue
        try:
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            for filename, target in [
                ("worker_tracking_data.csv", tracking),
                ("worker_movement_summary.csv", movement),
                ("restricted_zone_observations.csv", zones),
                ("risk_event_report.csv", events),
            ]:
                p = folder / filename
                if p.exists():
                    df = pd.read_csv(p)
                    if not df.empty:
                        target.append(df)
        except Exception as exc:
            st.warning(f"Could not load result folder {folder.name}: {exc}")

    def combine(items):
        return pd.concat(items, ignore_index=True) if items else pd.DataFrame()
    return combine(tracking), combine(movement), combine(zones), combine(events), summaries


st.sidebar.header("1. Upload videos")
uploads = st.sidebar.file_uploader(
    "Select one or more construction-site videos",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True,
    help="Videos are processed one at a time to limit memory usage.",
)

st.sidebar.header("2. Processing settings")
weights = st.sidebar.text_input("YOLO model weights", value="yolo11n.pt")
confidence = st.sidebar.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)
min_observations = st.sidebar.number_input("Minimum observations per worker track", 1, 100, 3)
device_choice = st.sidebar.selectbox("Inference device", ["Auto", "CPU", "GPU (CUDA)"])
save_annotated = st.sidebar.checkbox("Save annotated output video", value=True)

video_paths: list[Path] = []
video_metadata: dict[str, object] = {}
if uploads:
    for uploaded in uploads:
        try:
            path = persist_upload(uploaded)
            metadata, first_frame = inspect_video(path)
            video_paths.append(path)
            video_metadata[str(path)] = (metadata, first_frame)
        except Exception as exc:
            st.error(f"{uploaded.name}: {exc}")

zone_settings: dict[str, dict | None] = {}
if video_paths:
    st.subheader("3. Configure each video's restricted zone")
    st.caption("The preview is the first frame. Set a rectangle separately for each camera view; coordinates are percentages of that video's resolution.")
    for path in video_paths:
        metadata, first_frame = video_metadata[str(path)]
        with st.expander(f"{path.name} — {metadata.width} × {metadata.height}, {metadata.duration_seconds:.1f}s", expanded=len(video_paths) == 1):
            st.image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), caption="First frame — use this as a reference when configuring the zone", use_container_width=True)
            enabled = st.checkbox("Enable restricted-zone monitoring for this video", value=True, key=f"zone_enabled_{path.name}")
            if enabled:
                c1, c2, c3, c4 = st.columns(4)
                x1 = c1.slider("Left (%)", 0, 99, 40, key=f"zx1_{path.name}")
                y1 = c2.slider("Top (%)", 0, 99, 40, key=f"zy1_{path.name}")
                x2 = c3.slider("Right (%)", 1, 100, 70, key=f"zx2_{path.name}")
                y2 = c4.slider("Bottom (%)", 1, 100, 80, key=f"zy2_{path.name}")
                if x2 <= x1 or y2 <= y1:
                    st.error("Right must be greater than Left, and Bottom must be greater than Top.")
                    zone_settings[str(path)] = None
                else:
                    zone_settings[str(path)] = normalized_zone_to_pixels(metadata.width, metadata.height, x1, y1, x2, y2)
                    st.caption(f"Pixel zone: {zone_settings[str(path)]}")
                    preview = first_frame.copy()
                    z = zone_settings[str(path)]
                    cv2.rectangle(preview, (z["x1"], z["y1"]), (z["x2"], z["y2"]), (0, 0, 255), 3)
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Restricted-zone preview (red rectangle)", use_container_width=True)
            else:
                zone_settings[str(path)] = None

    if st.button("Process all uploaded videos", type="primary", use_container_width=True):
        if any(st.session_state.get(f"zone_enabled_{p.name}", True) and zone_settings.get(str(p)) is None for p in video_paths):
            st.error("Correct the restricted-zone coordinates or disable zone monitoring for the affected video.")
        else:
            try:
                model = load_model(weights.strip())
                if device_choice == "Auto":
                    import torch
                    device = 0 if torch.cuda.is_available() else "cpu"
                elif device_choice == "GPU (CUDA)":
                    import torch
                    if not torch.cuda.is_available():
                        st.error("CUDA is not available in this environment. Select CPU or enable a CUDA GPU.")
                        st.stop()
                    device = 0
                else:
                    device = "cpu"

                overall = st.progress(0, text="Starting batch...")
                status = st.empty()
                successes, failures = 0, []
                for index, path in enumerate(video_paths, start=1):
                    status.write(f"Processing {index}/{len(video_paths)}: **{path.name}**")
                    try:
                        result = process_video(
                            video_path=path,
                            model=model,
                            results_root=RESULTS_DIR,
                            confidence=confidence,
                            min_person_observations=int(min_observations),
                            zone=zone_settings.get(str(path)),
                            save_annotated_video=save_annotated,
                            device=device,
                        )
                        successes += 1
                        st.success(
                            f"Completed {path.name}: {result['summary']['unique_workers']} worker tracks, "
                            f"{result['summary']['event_count']} zone events."
                        )
                    except Exception as exc:
                        failures.append((path.name, str(exc)))
                        st.error(f"Failed {path.name}: {exc}")
                    overall.progress(index / len(video_paths), text=f"Finished {index} of {len(video_paths)} videos")
                status.empty()
                st.success(f"Batch finished: {successes} succeeded, {len(failures)} failed.")
                if failures:
                    st.dataframe(pd.DataFrame(failures, columns=["Video", "Error"]), use_container_width=True)
                st.rerun()
            except Exception as exc:
                st.exception(exc)

st.divider()
st.subheader("4. Processed video analytics")
tracking_df, movement_df, zone_df, events_df, summaries = load_results()
if not summaries:
    st.warning("No processed videos found yet. Upload one or more videos above and start processing.")
    st.stop()

summary_df = pd.DataFrame(summaries)
selected_ids = st.multiselect(
    "Filter dashboard by video",
    options=summary_df["video_id"].tolist(),
    default=summary_df["video_id"].tolist(),
    format_func=lambda value: summary_df.loc[summary_df.video_id.eq(value), "filename"].iloc[0],
)
selected_summary = summary_df[summary_df["video_id"].isin(selected_ids)].copy()
tracking_view = tracking_df[tracking_df["video_id"].isin(selected_ids)].copy() if not tracking_df.empty else tracking_df
movement_view = movement_df[movement_df["video_id"].isin(selected_ids)].copy() if not movement_df.empty else movement_df
zone_view = zone_df[zone_df["video_id"].isin(selected_ids)].copy() if not zone_df.empty else zone_df
events_view = events_df[events_df["video_id"].isin(selected_ids)].copy() if not events_df.empty else events_df

workers = int(tracking_view[["video_id", "track_id"]].drop_duplicates().shape[0]) if not tracking_view.empty else 0
observations = int(len(tracking_view))
zone_observations = int(len(zone_view))
event_count = int(len(events_view))
high_events = int(events_view["risk_level"].eq("High").sum()) if not events_view.empty else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Videos selected", len(selected_summary))
m2.metric("Tracked workers", workers)
m3.metric("Person observations", f"{observations:,}")
m4.metric("Zone events", event_count)
m5.metric("High rule-score events", high_events)

if not selected_summary.empty:
    st.markdown("#### Video processing summary")
    st.dataframe(selected_summary[[
        "filename", "processed_frames", "unique_workers", "person_observations",
        "zone_observations", "event_count", "high_risk_rule_events",
    ]], use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown("#### Risk event classification")
    if not events_view.empty:
        risk_counts = events_view["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Risk level").reset_index(name="Events")
        st.plotly_chart(px.bar(risk_counts, x="Risk level", y="Events", color="Risk level", title="Duration-rule events by severity"), use_container_width=True)
    else:
        st.info("No restricted-zone events in the selected videos.")
with right:
    st.markdown("#### Events by video")
    if not events_view.empty:
        names = summary_df.set_index("video_id")["filename"].to_dict()
        by_video = events_view.groupby("video_id").size().rename("Events").reset_index()
        by_video["Video"] = by_video["video_id"].map(names)
        st.plotly_chart(px.bar(by_video, x="Video", y="Events", title="Restricted-zone episodes per video"), use_container_width=True)
    else:
        st.info("No events available to compare.")

st.markdown("#### Worker trajectories")
if not tracking_view.empty and {"bottom_x", "bottom_y", "track_id"}.issubset(tracking_view.columns):
    trajectory = tracking_view.dropna(subset=["bottom_x", "bottom_y"]).copy()
    trajectory["worker_key"] = trajectory["video_id"].astype(str) + " / Track " + trajectory["track_id"].astype(str)
    names = summary_df.set_index("video_id")["filename"].to_dict()
    trajectory["video"] = trajectory["video_id"].map(names)
    fig = px.line(trajectory.sort_values(["video_id", "track_id", "frame"]), x="bottom_x", y="bottom_y", color="worker_key", hover_data=["video", "frame", "timestamp", "confidence"], title="Tracked person paths (pixel coordinates)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No worker trajectory records are available for the selected videos.")

st.markdown("#### Risk event report")
if not events_view.empty:
    st.dataframe(events_view, use_container_width=True, hide_index=True)
    st.download_button("Download combined risk-event CSV", events_view.to_csv(index=False).encode("utf-8"), "combined_risk_event_report.csv", "text/csv")
else:
    st.info("No risk events to display.")

with st.expander("Download and inspect individual video outputs"):
    for summary in selected_summary.to_dict("records"):
        folder = RESULTS_DIR / summary["video_id"]
        st.markdown(f"**{summary['filename']}**")
        for filename in ["worker_tracking_data.csv", "all_object_tracking_data.csv", "worker_movement_summary.csv", "restricted_zone_observations.csv", "risk_event_report.csv", "annotated_video.mp4"]:
            file_path = folder / filename
            if file_path.exists():
                st.download_button(
                    f"Download {filename}", file_path.read_bytes(),
                    file_name=f"{summary['video_id']}_{filename}",
                    key=f"download_{summary['video_id']}_{filename}",
                )

with st.expander("Methodology and limitations"):
    st.markdown("""
    - YOLO11n is used as a pretrained object detector; this project does not fine-tune it on a construction-specific dataset.
    - ByteTrack identities are local to each video. `video_id` + `track_id` identifies a track record across the batch.
    - Worker footpoints are estimated from the bottom-centre of each bounding box.
    - Restricted-zone checks use a manually configured rectangle in each video's image coordinates.
    - Movement and trajectory measurements are in pixels. They are not metres or calibrated physical speeds.
    - The current risk score is a transparent duration-based rule, not a learned ML prediction or accident probability.
    - This prototype analyzes uploaded recordings; it does not connect to live CCTV or guarantee hazard detection.
    """)

st.caption("Construction Safety Intelligence Prototype | Streamlit · OpenCV · YOLO11 · ByteTrack · Pandas · Plotly")
