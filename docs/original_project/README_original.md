# 🏗️ AI-Powered Construction Site Hazard Anticipation

### Video Understanding, Worker Trajectory Analysis, and Rule-Based Risk Monitoring

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/Object_Detection-YOLO11-green)
![Tracking](https://img.shields.io/badge/Object_Tracking-ByteTrack-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-purple)
![Project Status](https://img.shields.io/badge/Status-Prototype-yellow)

---

## 1. Project Overview

Construction sites are dynamic environments where workers, machinery, and equipment operate simultaneously. Monitoring these environments manually through video surveillance can be time-consuming and may make it difficult to review worker movements and identify potentially unsafe situations.

This project develops an AI-assisted construction safety monitoring prototype using computer vision and video analytics.

The system processes recorded construction-site footage, detects visible objects using a pretrained YOLO model, tracks their movement across consecutive frames using ByteTrack, and analyzes worker trajectories.

A manually configured restricted-zone monitoring mechanism identifies worker observations inside a predefined image region. Consecutive zone observations are grouped into intrusion episodes and assigned rule-based risk scores according to their duration.

The exported analytical results are presented through an interactive Streamlit dashboard.

### Key capabilities

- Object detection using pretrained YOLO11n.
- Multi-object tracking using ByteTrack.
- Worker identification through tracking IDs.
- Worker trajectory visualization.
- Movement feature engineering.
- Configurable restricted-zone monitoring.
- Intrusion episode identification.
- Duration-based risk classification.
- Interactive Streamlit dashboard.
- CSV-based analytical reporting.

**Project objective:** Demonstrate how computer vision and video analytics can support construction-site safety monitoring by transforming recorded footage into structured movement data and rule-based safety-event reports.

---

## 2. Business Problem

Construction sites involve continuous movement of workers and equipment. Traditional video surveillance primarily provides recorded footage, requiring safety personnel to manually review the recordings.

This creates several challenges:

1. Manual video inspection can be time-consuming.
2. Identifying individual workers across consecutive frames is difficult without tracking.
3. Worker movement patterns are not readily available in structured data.
4. Configurable restricted areas require continuous monitoring.
5. Reviewing and reporting potential zone intrusions can require additional effort.

### Proposed solution

The project introduces a computer-vision workflow that:

1. Detects visible objects in video frames.
2. Tracks detected objects across consecutive frames.
3. Extracts worker movement trajectories.
4. Identifies observations inside a configured restricted region.
5. Groups consecutive observations into intrusion episodes.
6. Assigns transparent duration-based risk scores.
7. Presents the results through an interactive dashboard.

### Business value

The prototype demonstrates a potential approach to reducing manual video-review effort and organizing safety-related observations into structured reports.

It is intended as a decision-support demonstration, not a replacement for professional safety supervision.

---

## 3. Project Objectives

| Objective | Implementation |
|---|---|
| Video processing | OpenCV |
| Object detection | YOLO11n |
| Object tracking | ByteTrack |
| Worker tracking | Unique tracking IDs |
| Movement analysis | NumPy and Pandas |
| Trajectory visualization | Matplotlib and Plotly |
| Restricted-zone monitoring | Coordinate-based rule |
| Intrusion identification | Consecutive frame grouping |
| Risk classification | Duration-based scoring |
| Interactive reporting | Streamlit |

---

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| YOLO11n | Pretrained object detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video reading and frame processing |
| PyTorch | Deep-learning inference and GPU support |
| NumPy | Numerical computations |
| Pandas | Data manipulation and feature engineering |
| Matplotlib | Trajectory and analytical visualization |
| Plotly | Interactive dashboard charts |
| Streamlit | Dashboard development |
| Scikit-learn | Optional evaluation utilities |
| Google Colab | Notebook development and GPU execution |
| VS Code | Local development and dashboard execution |
| Git & GitHub | Version control and project documentation |

---

## 5. System Architecture

The project follows a modular video analytics pipeline.

```text
                INPUT VIDEO
                     |
                     v
             VIDEO PROCESSING
                 OpenCV
                     |
                     v
           OBJECT DETECTION
                YOLO11n
                     |
                     v
           MULTI-OBJECT TRACKING
                ByteTrack
                     |
                     v
          TRACKING DATA GENERATION
       Bounding Boxes | Track IDs | Timestamp
                     |
                     v
          DATA AUDIT & PREPROCESSING
                     |
                     v
          WORKER MOVEMENT ANALYSIS
        Trajectory | Displacement | Features
                     |
                     v
          RESTRICTED ZONE MONITORING
           Manual Coordinate Rectangle
                     |
                     v
           INTRUSION EVENT GROUPING
                     |
                     v
          RULE-BASED RISK SCORING
              Low | Medium | High
                     |
                     v
             CSV EXPORT MODULE
                     |
                     v
             STREAMLIT DASHBOARD
                     |
                     v
          INTERACTIVE SAFETY REPORT
```

### Pipeline explanation

**Stage 1 – Video input**

A recorded video is uploaded and inspected using OpenCV. Video properties such as resolution, frame count, FPS, and duration are extracted.

**Stage 2 – Object detection**

YOLO11n identifies objects in individual frames and generates bounding boxes, class labels, and confidence scores.

**Stage 3 – Object tracking**

ByteTrack associates detections across consecutive frames and assigns tracking IDs.

**Stage 4 – Movement analysis**

Bounding-box coordinates are used to calculate worker positions, pixel displacement, and trajectory information.

**Stage 5 – Restricted-zone monitoring**

A manually configured rectangular region is used to identify worker footpoints located inside the sample zone.

**Stage 6 – Risk-event identification**

Consecutive in-zone observations are grouped into intrusion episodes. Each episode receives a duration-based score and risk label.

**Stage 7 – Dashboard and reporting**

The generated CSV files are loaded into Streamlit to display monitoring metrics, charts, trajectories, and event reports.

---

## 6. Computer Vision Methodology

### 6.1 YOLO Object Detection

The project uses the pretrained YOLO11n model from Ultralytics.

YOLO performs object detection by identifying:

- Bounding-box coordinates.
- Object class IDs.
- Object class names.
- Detection confidence scores.

The model is loaded using:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

Detection is performed with a confidence threshold of 0.25.

```python
results = model.predict(
    source=frame,
    conf=0.25,
    device=device,
    verbose=False
)
```

**Why YOLO?**

YOLO provides a practical single-stage object detection approach that can identify multiple objects in video frames.

The pretrained model is used without construction-specific fine-tuning in the current implementation.

### 6.2 ByteTrack Object Tracking

Object detection identifies objects in individual frames, but it does not maintain their identities across time.

ByteTrack addresses this by associating detections across consecutive frames.

```python
tracking_results = model.track(
    source=str(VIDEO_PATH),
    tracker="bytetrack.yaml",
    conf=0.25,
    device=device,
    stream=True,
    persist=True,
    verbose=False
)
```

The tracking process generates:

- Frame number.
- Timestamp.
- Track ID.
- Object class.
- Bounding-box coordinates.
- Detection confidence.

Tracking IDs allow the project to construct individual object histories.

**Important:** Tracking IDs are temporary identifiers within a video-processing run. They are not permanent worker identities.

### 6.3 Worker Trajectory Analysis

The bottom-centre point of each bounding box is used as the reference position for movement analysis.

The coordinates are calculated as:

```python
bottom_x = (x1 + x2) / 2
bottom_y = y2
```

The displacement between consecutive observations is calculated using Euclidean distance:

\[
D = \sqrt{(\Delta x)^2 + (\Delta y)^2}
\]

Where:

- Δx = Change in horizontal pixel position.
- Δy = Change in vertical pixel position.
- D = Pixel displacement.

Movement features include:

- Horizontal displacement.
- Vertical displacement.
- Frame gap.
- Euclidean displacement.
- Displacement per frame.
- Track duration.
- Average detection confidence.

These measurements describe image-coordinate movement, not actual physical distance or walking speed.

---

## 7. Restricted-Zone Monitoring

The project implements a coordinate-based restricted-zone monitoring mechanism.

A rectangular region is manually configured using four coordinates:

```python
ZONE_X1 = 2250
ZONE_Y1 = 2135
ZONE_X2 = 2380
ZONE_Y2 = 2160
```

The bottom-centre point of each worker bounding box is checked against this rectangle.

The point is considered inside the zone when:

\[
X_1 \leq X \leq X_2
\]

\[
Y_1 \leq Y \leq Y_2
\]

Python implementation:

```python
person_df["inside_restricted_zone"] = (
    (person_df["bottom_x"] >= ZONE_X1) &
    (person_df["bottom_x"] <= ZONE_X2) &
    (person_df["bottom_y"] >= ZONE_Y1) &
    (person_df["bottom_y"] <= ZONE_Y2)
)
```

Observations satisfying this condition are extracted for further analysis.

**Note:** The restricted region is manually configured for demonstration. Its coordinates must be calibrated to the actual scene before it can represent a meaningful site-specific safety boundary.

---

## 8. Intrusion Event Identification and Risk Scoring

Individual zone observations are grouped into intrusion episodes based on consecutive frame observations.

Each episode contains:

- Track ID.
- Event ID.
- Entry frame.
- Exit frame.
- Entry timestamp.
- Exit timestamp.
- Number of observations.
- Average confidence.
- Duration.
- Risk score.
- Risk level.

### Duration calculation

\[
Duration = Exit\ Time - Entry\ Time
\]

### Rule-based risk classification

The current prototype applies the following scoring mechanism:

| Duration | Risk Score | Risk Level |
|---|---:|---|
| Less than 2 seconds | 25 | Low |
| 2 to less than 5 seconds | 50 | Medium |
| 5 seconds or more | 80 | High |

Python implementation:

```python
events["risk_score"] = np.select(
    [
        events["duration_seconds"] >= 5,
        events["duration_seconds"] >= 2
    ],
    [80, 50],
    default=25
)

events["risk_level"] = np.select(
    [
        events["risk_score"] >= 70,
        events["risk_score"] >= 40
    ],
    ["High", "Medium"],
    default="Low"
)
```

### Important distinction

These risk scores are deterministic, duration-based rules.

They are not:

- Machine-learning predictions.
- Accident probabilities.
- Validated safety thresholds.
- Evidence that an accident will occur.

The scoring mechanism is included to demonstrate transparent event classification.

---

## 9. Streamlit Dashboard

The project includes an interactive dashboard developed using Streamlit and Plotly.

The dashboard consumes CSV files exported from the Google Colab notebook.

### Dashboard components

**1. Safety Monitoring Overview**

Displays five monitoring metrics:

- Tracked workers.
- Person observations.
- Zone observations.
- Intrusion episodes.
- High-rule-score events.

**2. Risk Event Classification**

A Plotly bar chart displaying the number of events classified as Low, Medium, and High.

**3. Zone Observations by Track**

Visualizes the number of observations inside the configured sample zone for each tracking ID.

**4. Worker Movement Trajectories**

Displays worker movement paths using bottom-centre pixel coordinates.

**5. Risk Event Report**

Provides a structured table containing event information, timestamps, duration, risk score, and risk classification.

**6. Worker Movement Summary**

Displays the exported movement summary when available.

### Dashboard preview

The dashboard can be launched locally using:

```bash
streamlit run app.py
```

The default application address is:

```text
http://localhost:8501
```

---

## 10. Project Results

The following figures represent the documented run in the uploaded notebook. Different videos, model versions, and configurations may produce different results.

### Tracking results

| Metric | Observed Result |
|---|---:|
| Total tracked-object observations | 837 |
| Total columns in tracking dataset | 14 |
| Person observations | 824 |
| Substantial person tracks selected for movement analysis | 2 |
| Person tracking IDs | 1 and 2 |

The tracking audit also identified a short additional track and other predicted object classes. These were not treated as confirmed construction-site objects.

### Restricted-zone results

| Metric | Observed Result |
|---|---:|
| Person observations analyzed | 824 |
| Observations inside sample zone | 508 |
| Intrusion episodes | 2 |
| High-risk episodes | 1 |
| Low-risk episodes | 1 |

### Intrusion event report

| Track ID | Event ID | Duration | Risk Score | Risk Level |
|---:|---:|---:|---:|---|
| 2 | 1 | 10 seconds | 80 | High |
| 2 | 2 | 0.12 seconds | 25 | Low |

These events are sample-zone matches generated by the configured coordinate rectangle and duration-based rules. They are not independently verified safety violations.

---

## 11. Dataset and Input Requirements

The notebook accepts a recorded video uploaded through Google Colab.

Supported video formats:

- MP4
- AVI
- MOV
- MKV

The current implementation selects the first matching video file in the configured video directory.

It processes one selected video per run.

### Input requirements

For meaningful analysis, the footage should ideally contain:

- Clearly visible people.
- Sufficient video resolution.
- Consecutive frames showing object movement.
- A reasonably stable camera perspective.
- A visible region suitable for configuring a sample zone.

The current notebook does not include an independently labelled accident or near-miss dataset.

No dataset-based accident-prediction evaluation has been performed.

---

## 12. Exported Project Outputs

The notebook generates the following output files:

| Filename | Description |
|---|---|
| `worker_tracking_data.csv` | Frame-level person tracking records and movement features |
| `worker_movement_summary.csv` | Aggregated worker movement statistics |
| `restricted_zone_observations.csv` | Worker observations inside the configured zone |
| `risk_event_report.csv` | Grouped intrusion episodes and risk classifications |
| `worker_trajectories.png` | Worker trajectory visualization |
| `construction_safety_project_results.zip` | Compressed output package |

The CSV files are used as inputs to the Streamlit dashboard.

---

## 13. Repository Structure

Recommended GitHub repository structure:

```text
AI-Construction-Safety-Monitoring/
│
├── app.py
│
├── construction_hazard_project.ipynb
│
├── construction_safety_outputs/
│   ├── worker_tracking_data.csv
│   ├── worker_movement_summary.csv
│   ├── restricted_zone_observations.csv
│   └── risk_event_report.csv
│
├── reports/
│   └── worker_trajectories.png
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

The original video, model weights, and large generated files can be kept outside the Git repository.

---

## 14. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Google Colab account.
- VS Code or another Python IDE.
- Git (optional, for version control).

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate into the project directory:

```bash
cd YOUR_REPOSITORY
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the output directory

Create the following folder:

```text
construction_safety_outputs/
```

Place the four required CSV files inside it.

Alternatively, upload them directly through the Streamlit sidebar.

### Step 5: Launch the dashboard

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 15. Requirements

Create a `requirements.txt` file with the following dependencies:

```text
streamlit
pandas
numpy
plotly
matplotlib
opencv-python-headless
ultralytics
torch
scikit-learn
```

For the Streamlit dashboard alone, the essential dependencies are:

```text
streamlit
pandas
plotly
```

The remaining packages are used by the notebook's video-processing and analysis workflow.

For reproducible deployment, pin the tested package versions and use a compatible PyTorch installation for the target hardware.

---

## 16. How to Run the Complete Project

The project has two main execution components.

### Component A: Video Analytics Notebook

1. Open Google Colab.
2. Upload `construction_hazard_project.ipynb`.
3. Select a GPU runtime if available.
4. Install the required dependencies.
5. Upload the construction-site video.
6. Execute the notebook cells sequentially.
7. Review object detection and tracking results.
8. Inspect movement trajectories.
9. Configure the restricted-zone coordinates.
10. Execute the event identification and scoring cells.
11. Export the CSV and PNG outputs.
12. Download `construction_safety_project_results.zip`.

### Component B: Streamlit Dashboard

1. Extract the downloaded ZIP file.
2. Place the CSV files inside `construction_safety_outputs/`, or keep them ready for manual upload.
3. Open the Streamlit project folder in VS Code.
4. Activate the Python environment.
5. Install the dashboard dependencies.
6. Run:

```bash
streamlit run app.py
```

7. Upload the required CSV files if they are not available in the local output directory.
8. Explore the monitoring dashboard.

---

## 17. Evaluation and Validation

The current project demonstrates a functioning computer-vision and rule-based analytics workflow.

However, a working pipeline should not be confused with a validated predictive model.

| Component | Current Evaluation | Further Validation Required |
|---|---|---|
| Object detection | Visual inspection of predictions | Precision, recall, mAP using labelled frames |
| Object tracking | Track histories and visual inspection | IDF1, HOTA, identity-switch analysis |
| Trajectory analysis | Pixel-coordinate movement plots | Camera calibration and trajectory verification |
| Zone monitoring | Coordinate-based zone matching | Expert-verified zone boundaries |
| Risk classification | Rule-based event scores | Independently labelled safety events |
| Accident prediction | Not implemented | Labelled temporal data and predictive evaluation |

Metrics such as precision, recall, F1-score, PR-AUC, and warning lead time should only be reported after establishing suitable ground-truth labels and an appropriate evaluation methodology.

---

## 18. Limitations

The current implementation has several limitations:

1. **Single-video processing:** The notebook processes one selected video per execution.
2. **Pretrained detection model:** YOLO11n has not been fine-tuned on construction-specific footage.
3. **Manual zone configuration:** Restricted-zone boundaries are manually specified using image coordinates.
4. **Rule-based risk scores:** Risk classification depends on predefined duration thresholds.
5. **Pixel-based movement:** Coordinates do not represent actual distances or physical speeds.
6. **No validated machinery interaction detection:** Worker–machinery proximity and collision-risk detection are not implemented.
7. **No accident prediction model:** The project does not estimate the probability of future accidents.
8. **No live CCTV integration:** The Streamlit application displays exported notebook results rather than processing live camera streams.
9. **Limited validation:** There is no independently labelled accident dataset for evaluating predictive performance.

---

## 19. Future Enhancements

The project can be extended into a more comprehensive construction safety monitoring platform.

### Computer Vision Enhancements

- Fine-tune YOLO on construction-specific datasets.
- Improve detection of workers, helmets, safety vests, and machinery.
- Implement PPE compliance monitoring.
- Improve tracking reliability in crowded environments.

### Advanced Safety Analytics

- Worker–machinery proximity monitoring.
- Unsafe movement and interaction detection.
- Dynamic restricted-zone configuration.
- Temporal analysis of movement patterns.
- Integration of independently labelled near-miss events.

### Machine Learning and Deep Learning

- Develop supervised risk-classification models using labelled safety-event data.
- Explore LSTM or other temporal models for sequential movement analysis.
- Evaluate prediction lead time for genuinely future-oriented warning tasks.
- Compare rule-based baselines with trained predictive models.

### Deployment Enhancements

- Multiple CCTV video-stream integration.
- Real-time video processing.
- Automated safety alerts.
- Centralized monitoring dashboard.
- Cloud-based storage and reporting.
- Role-based access for safety personnel.

---

## 20. Business Applications

A mature and independently validated version of this system could support:

- Construction-site safety monitoring.
- Restricted-area access monitoring.
- Video-based safety inspections.
- Worker movement analysis.
- Safety-event documentation.
- Historical safety reporting.
- Construction-site operational analytics.

The current prototype demonstrates the technical foundation for these applications but does not establish production readiness.

---

## 21. Key Learning Outcomes

This project provided practical experience in:

- Applying pretrained deep-learning models to video data.
- Understanding object detection versus object tracking.
- Processing video frames using OpenCV.
- Implementing YOLO inference.
- Using ByteTrack for multi-object tracking.
- Extracting structured tracking information.
- Performing movement feature engineering.
- Visualizing object trajectories.
- Implementing coordinate-based spatial rules.
- Developing event-grouping logic.
- Designing transparent rule-based scoring.
- Building interactive analytical dashboards using Streamlit.
- Connecting computer-vision outputs with business-oriented reporting.

---

## 22. Conclusion

The AI-Powered Construction Site Hazard Anticipation project demonstrates an integrated computer-vision workflow for analyzing construction-site video footage.

By combining pretrained YOLO11n object detection with ByteTrack object tracking, the system extracts object-level observations and maintains tracking identities across consecutive frames.

The resulting tracking information is used to analyze worker movement, visualize trajectories, and identify observations within a manually configured restricted zone.

Consecutive zone observations are grouped into intrusion episodes and assigned duration-based risk scores. These analytical outputs are exported into structured CSV files and presented through an interactive Streamlit dashboard.

Although the current implementation is a prototype rather than a validated accident-prediction system, it establishes a foundation for extending video surveillance into structured, AI-assisted construction safety analytics.

---

## 23. Author and Project Information

**Project Title:** AI-Powered Construction Site Hazard Anticipation

**Domain:** Artificial Intelligence | Computer Vision | Deep Learning | Data Analytics

**Primary Technologies:** Python, YOLO11n, ByteTrack, OpenCV, Pandas, NumPy, Streamlit, Plotly

**Development Environment:** Google Colab and VS Code

**Project Type:** Individual Internship Project

**Project Status:** Prototype

---

## Disclaimer

This project is developed for educational, research, and demonstration purposes.

The system's detection results, tracking information, restricted-zone observations, and rule-based risk scores are not certified safety assessments.

The prototype must not be used as the sole basis for construction-site safety decisions, accident prevention, or regulatory compliance.

Professional safety supervision and independently validated monitoring systems remain necessary.
