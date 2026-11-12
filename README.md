# AI-Powered Construction Site Hazard Anticipation

## Multi-video Streamlit prototype (refactored version)

This version preserves the original workflow—pretrained YOLO object detection, ByteTrack multi-object tracking, worker trajectory analysis, manually configured restricted zones, duration-based event scoring, and Streamlit reporting—while extending it from one preprocessed video to multiple uploaded videos processed directly by the application.

## What changed from the original project

- Multiple MP4, AVI, MOV, and MKV uploads are accepted in one batch.
- Videos are processed sequentially to reduce peak memory use.
- Every video gets a unique `video_id`; ByteTrack `track_id` values are treated as local to that video.
- All person tracks are selected dynamically; no worker IDs are hardcoded.
- Restricted zones are configured separately for each video as percentages of its frame dimensions, with a red-rectangle preview.
- The processing code is separated into `safety_pipeline.py` and can be reused outside Streamlit.
- Per-video CSVs and JSON summaries are saved under `app_data/results/<video_id>/`.
- An optional annotated MP4 is generated for each video.
- The dashboard supports individual-video filtering and combined batch analytics.

## Architecture

```text
Multiple uploaded videos
        |
        v
Streamlit upload + per-video zone configuration
        |
        v
safety_pipeline.process_video(video)
        |
        +--> OpenCV video reader
        +--> YOLO11 pretrained detector
        +--> ByteTrack tracker (independent run per video)
        +--> Worker observations + trajectories
        +--> Restricted-zone point-in-rectangle rule
        +--> Consecutive observation episode grouping
        +--> Duration-based prototype risk score
        |
        v
Per-video CSV / JSON / optional annotated MP4
        |
        v
Combined Streamlit dashboard and downloads
```

## Project files

- `app.py` — Streamlit UI, multi-upload orchestration, per-video configuration, dashboard.
- `safety_pipeline.py` — reusable video inspection, YOLO/ByteTrack inference, movement features, zone events, exports.
- `requirements.txt` — Python dependencies.
- `app_data/uploads/` — staged uploaded videos (ignored by Git).
- `app_data/results/` — per-video outputs (ignored by Git).

## Run locally

Use Python 3.10 or 3.11 in a virtual environment. A CUDA-enabled PyTorch installation is recommended for GPU inference; install the PyTorch build appropriate for your machine before installing the remaining packages if needed.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

On first run, Ultralytics may download `yolo11n.pt`. Ensure the machine has internet access for that initial model download, or provide a local weights path in the sidebar.

## How to use

1. Start the Streamlit app.
2. Upload one or more recorded construction-site videos.
3. Inspect each video's first-frame preview.
4. Enable or disable restricted-zone monitoring separately for each video.
5. If enabled, adjust the red rectangle using the left/top/right/bottom percentage sliders. The red preview shows the configured zone.
6. Choose confidence threshold, minimum person-track observations, inference device, and annotated-video output.
7. Click **Process all uploaded videos**. Videos run sequentially.
8. Filter the dashboard by video and download the individual or combined CSV reports.

## Output files per video

- `video_summary.json` — source metadata, counts, zone coordinates, processing metadata.
- `all_object_tracking_data.csv` — all tracked object observations.
- `worker_tracking_data.csv` — person observations and movement fields.
- `worker_movement_summary.csv` — track-level movement summary.
- `restricted_zone_observations.csv` — worker observations inside the configured rectangle.
- `risk_event_report.csv` — grouped restricted-zone episodes and rule scores.
- `annotated_video.mp4` — optional annotated output.

## Important methodology notes

- YOLO11n is pretrained; this repository does not fine-tune it on a construction-specific dataset.
- ByteTrack track IDs are local to each source video. Use `(video_id, track_id)` as the combined identity key.
- A worker footpoint is approximated by the bottom-centre of the detection box.
- Trajectory and movement values are in pixels. They are not physical distances or calibrated speeds.
- A restricted-zone event is based on a manually configured rectangular image region and observed worker footpoints.
- The initial risk score deliberately preserves the original notebook's duration rules: under 2 seconds = 25 / Low; 2 to under 5 seconds = 50 / Medium; 5 seconds or more = 80 / High.
- These values are heuristic labels, not trained ML predictions, accident probabilities, or validated safety thresholds.
- The system processes uploaded recordings; it is not a live CCTV monitoring service.
- Detection, tracking, and event quality must be evaluated against manually annotated ground truth before making operational safety claims.

## Current limitations and next engineering steps

1. Add an interactive polygon ROI editor rather than a rectangular zone.
2. Add PPE-specific detection only after selecting a suitable labelled dataset and validating a detector; the pretrained COCO model does not provide reliable hard-hat/vest compliance detection by itself.
3. Add worker–vehicle proximity rules with camera calibration and carefully defined event labels.
4. Build a labelled evaluation set and report detection precision/recall, tracking metrics, and event-level precision/recall.
5. Train a predictive risk model only after collecting sufficient labelled historical examples; do not treat heuristic scores as ML labels without validation.
6. For larger batches, move long-running processing to a background job queue and store metadata in SQLite/PostgreSQL.

## Hardware guidance

Video inference can be computationally expensive. Process a small batch first and monitor memory, GPU utilization, and per-video processing time. For an 8 GB RAM laptop, prefer short clips, YOLO11n, sequential processing, and an external CUDA GPU/Colab runtime when available. The application does not load every video into the model at once, but Streamlit's upload widget still stages uploaded file bytes in the application process.
