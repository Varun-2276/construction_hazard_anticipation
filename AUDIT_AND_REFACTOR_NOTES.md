# Audit and refactor notes

## Source material reviewed

- `construction_hazard_project_story_notebook (1).ipynb` (87 cells)
- Project ZIP: `app.py`, `README.md`, original notebook, PowerPoint, abstract PDF, and exported CSV/PNG files

## Preserved implementation

- Pretrained YOLO11n inference
- Ultralytics ByteTrack tracking
- Frame/time and bounding-box extraction
- Bottom-centre footpoint calculation
- Per-track movement and trajectory analysis
- Manually configured restricted-zone rectangle
- Intrusion episode grouping
- Original duration-based rule thresholds (2 seconds and 5 seconds)
- CSV-based outputs and Plotly/Streamlit reporting concept

## Issues found in original code

1. The notebook selected only track IDs 1 and 2 for person analysis. This fails for arbitrary videos and is replaced by dynamic person-track selection with a configurable minimum observation count.
2. Restricted-zone pixel coordinates were hardcoded for one camera view. The new app configures each video's zone independently using normalized coordinates and displays a preview.
3. Track IDs are not globally unique across videos. The refactor adds a video_id to all output rows; use `(video_id, track_id)` as the composite key.
4. The original Streamlit app loaded exported CSVs only and did not process videos. The new app accepts and processes uploaded videos.
5. The original notebook's risk score is duration-based, not a trained predictive model. This is preserved transparently and documented as a heuristic.
6. Movement features were calculated only for manually selected tracks. The refactor computes them for all eligible person tracks.
7. The original event grouping split on every non-consecutive frame. The refactor keeps this strict default, with a configurable gap tolerance in the pipeline helper.

## Refactor validation performed

- Python syntax compilation passed for `app.py` and `safety_pipeline.py`.
- Synthetic DataFrame test passed for movement feature generation.
- Synthetic restricted-zone test passed for point-in-rectangle selection and event grouping.
- Normalized-to-pixel zone conversion test passed.

## Not yet validated

- Full YOLO inference against real videos in this environment (requires the Ultralytics/PyTorch runtime and model weights).
- GPU/CPU performance on the user's machine.
- End-to-end Streamlit browser interaction.
- Detection/tracking accuracy against manually labelled ground truth.
