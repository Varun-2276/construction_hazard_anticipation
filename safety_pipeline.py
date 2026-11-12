"""Reusable, video-independent construction safety video processing pipeline.

The pipeline preserves the original prototype's YOLO + ByteTrack + trajectory
+ restricted-zone + duration-rule workflow, while removing single-video and
hardcoded-track assumptions.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

TRACK_COLUMNS = [
    "video_id", "frame", "timestamp", "track_id", "class_id", "class_name",
    "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y",
    "bottom_x", "bottom_y",
]
PERSON_COLUMNS = TRACK_COLUMNS + [
    "delta_x", "delta_y", "frame_gap", "displacement_pixels",
    "elapsed_seconds", "speed_px_per_second", "inside_restricted_zone",
]
MOVEMENT_COLUMNS = [
    "video_id", "track_id", "observations", "start_frame", "end_frame",
    "start_time", "end_time", "total_path_pixels", "net_x_displacement",
    "net_y_displacement", "average_speed_px_per_second",
    "maximum_speed_px_per_second", "average_confidence",
]
ZONE_COLUMNS = PERSON_COLUMNS
EVENT_COLUMNS = [
    "video_id", "event_id", "track_id", "entry_frame", "exit_frame",
    "entry_time", "exit_time", "observations", "avg_confidence",
    "duration_seconds", "risk_score", "risk_level",
]


@dataclass
class VideoMetadata:
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def inspect_video(video_path: str | Path) -> tuple[VideoMetadata, np.ndarray]:
    """Return video metadata and its first frame (BGR)."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {path.name}")
    ok, first_frame = cap.read()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if not ok or first_frame is None:
        raise ValueError(f"No readable first frame found in: {path.name}")
    if fps <= 0:
        raise ValueError(f"Video has invalid FPS metadata: {path.name}")
    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoMetadata(path.name, width, height, fps, frame_count, duration), first_frame


def normalized_zone_to_pixels(
    width: int, height: int, x1_pct: float, y1_pct: float,
    x2_pct: float, y2_pct: float,
) -> dict[str, int]:
    """Convert normalized 0..100 zone coordinates into image pixels."""
    if not (0 <= x1_pct < x2_pct <= 100 and 0 <= y1_pct < y2_pct <= 100):
        raise ValueError("Zone coordinates must satisfy x1 < x2 and y1 < y2 within 0–100%.")
    return {
        "x1": int(round(width * x1_pct / 100)),
        "y1": int(round(height * y1_pct / 100)),
        "x2": int(round(width * x2_pct / 100)),
        "y2": int(round(height * y2_pct / 100)),
    }


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def engineer_movement_features(
    person_df: pd.DataFrame, video_fps: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate image-coordinate movement per person track."""
    if person_df.empty:
        return _empty_frame(PERSON_COLUMNS), _empty_frame(MOVEMENT_COLUMNS)

    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for track_id, group in person_df.groupby("track_id", sort=False):
        g = group.sort_values("frame").copy()
        g["delta_x"] = g["bottom_x"].diff()
        g["delta_y"] = g["bottom_y"].diff()
        g["frame_gap"] = g["frame"].diff()
        g["displacement_pixels"] = np.hypot(g["delta_x"], g["delta_y"])
        g["elapsed_seconds"] = g["timestamp"].diff()
        g["speed_px_per_second"] = (
            g["displacement_pixels"] / g["elapsed_seconds"].replace(0, np.nan)
        )
        g["speed_px_per_second"] = g["speed_px_per_second"].replace(
            [np.inf, -np.inf], np.nan
        )
        rows.append(g)
        summaries.append({
            "video_id": g["video_id"].iloc[0],
            "track_id": int(track_id),
            "observations": int(len(g)),
            "start_frame": int(g["frame"].min()),
            "end_frame": int(g["frame"].max()),
            "start_time": float(g["timestamp"].min()),
            "end_time": float(g["timestamp"].max()),
            "total_path_pixels": float(g["displacement_pixels"].sum(skipna=True)),
            "net_x_displacement": float(g["bottom_x"].iloc[-1] - g["bottom_x"].iloc[0]),
            "net_y_displacement": float(g["bottom_y"].iloc[-1] - g["bottom_y"].iloc[0]),
            "average_speed_px_per_second": float(g["speed_px_per_second"].mean()),
            "maximum_speed_px_per_second": float(g["speed_px_per_second"].max()),
            "average_confidence": float(g["confidence"].mean()),
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries, columns=MOVEMENT_COLUMNS)


def identify_zone_events(
    person_df: pd.DataFrame,
    zone: Optional[dict[str, int]],
    video_fps: float,
    gap_tolerance_frames: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find footpoint observations in a rectangle and group them into episodes.

    With gap_tolerance_frames=0 (default), only consecutive in-zone frames are
    merged, matching the original notebook's strict episode grouping.
    """
    if person_df.empty:
        return _empty_frame(ZONE_COLUMNS), _empty_frame(EVENT_COLUMNS)
    if zone is None:
        people = person_df.copy()
        people["inside_restricted_zone"] = False
        return _empty_frame(ZONE_COLUMNS), _empty_frame(EVENT_COLUMNS)

    people = person_df.copy()
    people["inside_restricted_zone"] = (
        people["bottom_x"].between(zone["x1"], zone["x2"], inclusive="both")
        & people["bottom_y"].between(zone["y1"], zone["y2"], inclusive="both")
    )
    zone_df = people.loc[people["inside_restricted_zone"]].copy()
    if zone_df.empty:
        return zone_df.reindex(columns=ZONE_COLUMNS), _empty_frame(EVENT_COLUMNS)

    event_rows: list[dict[str, Any]] = []
    event_counter = 0
    for track_id, group in zone_df.groupby("track_id", sort=False):
        g = group.sort_values("frame").copy()
        # Split when the frame gap exceeds the configured tolerance + 1.
        new_episode = g["frame"].diff().fillna(1).gt(gap_tolerance_frames + 1)
        g["episode_number"] = new_episode.cumsum().astype(int)
        for _, episode in g.groupby("episode_number", sort=True):
            event_counter += 1
            entry_frame = int(episode["frame"].min())
            exit_frame = int(episode["frame"].max())
            entry_time = float(episode["timestamp"].min())
            exit_time = float(episode["timestamp"].max())
            # A one-frame observation represents at least one frame interval.
            duration = max(exit_time - entry_time, 1.0 / video_fps)
            score = 80 if duration >= 5 else (50 if duration >= 2 else 25)
            level = "High" if score >= 70 else ("Medium" if score >= 40 else "Low")
            event_rows.append({
                "video_id": str(episode["video_id"].iloc[0]),
                "event_id": f"E{event_counter:04d}",
                "track_id": int(track_id),
                "entry_frame": entry_frame,
                "exit_frame": exit_frame,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "observations": int(len(episode)),
                "avg_confidence": float(episode["confidence"].mean()),
                "duration_seconds": float(duration),
                "risk_score": int(score),
                "risk_level": level,
            })
    return zone_df.reindex(columns=ZONE_COLUMNS), pd.DataFrame(event_rows, columns=EVENT_COLUMNS)


def process_video(
    video_path: str | Path,
    model: Any,
    results_root: str | Path,
    confidence: float = 0.25,
    min_person_observations: int = 3,
    zone: Optional[dict[str, int]] = None,
    save_annotated_video: bool = True,
    device: str | int = "cpu",
) -> dict[str, Any]:
    """Run YOLO + ByteTrack and export one video's independent result bundle."""
    path = Path(video_path)
    metadata, _ = inspect_video(path)
    video_id = f"{re.sub(r'[^A-Za-z0-9_-]+', '_', path.stem)[:40]}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(results_root) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated_video.mp4"
    writer = None
    rows: list[dict[str, Any]] = []
    processed_frames = 0

    try:
        # A separate track() call is made for each source video. persist=False
        # avoids intentionally carrying tracker state between independent clips.
        stream = model.track(
            source=str(path), tracker="bytetrack.yaml", conf=confidence,
            device=device, stream=True, persist=False, verbose=False,
        )
        for frame_idx, result in enumerate(stream):
            processed_frames += 1
            original = result.orig_img
            if save_annotated_video and original is not None:
                if writer is None:
                    h, w = original.shape[:2]
                    writer = cv2.VideoWriter(
                        str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        metadata.fps, (w, h),
                    )
                annotated = result.plot()
                writer.write(annotated)

            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            for coords, track_id, class_id, conf in zip(xyxy, ids, classes, confidences):
                x1, y1, x2, y2 = [float(v) for v in coords]
                class_name = model.names[int(class_id)]
                rows.append({
                    "video_id": video_id,
                    "frame": int(frame_idx),
                    "timestamp": float(frame_idx / metadata.fps),
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "confidence": float(conf),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                    "bottom_x": (x1 + x2) / 2,
                    "bottom_y": y2,
                })
    finally:
        if writer is not None:
            writer.release()

    tracks_df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    if not tracks_df.empty:
        person_all = tracks_df.loc[tracks_df["class_name"].str.lower().eq("person")].copy()
        counts = person_all.groupby("track_id").size()
        valid_ids = counts[counts >= int(min_person_observations)].index
        person_all = person_all[person_all["track_id"].isin(valid_ids)].copy()
    else:
        person_all = _empty_frame(TRACK_COLUMNS)

    person_df, movement_df = engineer_movement_features(person_all, metadata.fps)
    zone_df, events_df = identify_zone_events(person_df, zone, metadata.fps)

    # Retain the zone flag on every worker observation for downstream analysis.
    if zone is None:
        person_df["inside_restricted_zone"] = False
    else:
        person_df["inside_restricted_zone"] = (
            person_df["bottom_x"].between(zone["x1"], zone["x2"], inclusive="both")
            & person_df["bottom_y"].between(zone["y1"], zone["y2"], inclusive="both")
        )
    for name, frame in [
        ("worker_tracking_data.csv", person_df),
        ("all_object_tracking_data.csv", tracks_df),
        ("worker_movement_summary.csv", movement_df),
        ("restricted_zone_observations.csv", zone_df),
        ("risk_event_report.csv", events_df),
    ]:
        frame.to_csv(output_dir / name, index=False)

    summary = {
        "video_id": video_id,
        "filename": path.name,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "width": metadata.width,
        "height": metadata.height,
        "fps": metadata.fps,
        "source_frame_count": metadata.frame_count,
        "source_duration_seconds": metadata.duration_seconds,
        "processed_frames": processed_frames,
        "tracked_object_observations": int(len(tracks_df)),
        "unique_object_tracks": int(tracks_df["track_id"].nunique()) if not tracks_df.empty else 0,
        "unique_workers": int(person_df["track_id"].nunique()) if not person_df.empty else 0,
        "person_observations": int(len(person_df)),
        "zone_observations": int(len(zone_df)),
        "event_count": int(len(events_df)),
        "high_risk_rule_events": int((events_df["risk_level"] == "High").sum()) if not events_df.empty else 0,
        "zone": zone,
        "risk_method": "duration-based prototype rule; not a trained prediction",
        "annotated_video": annotated_path.name if save_annotated_video and annotated_path.exists() else None,
    }
    (output_dir / "video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "output_dir": str(output_dir)}
