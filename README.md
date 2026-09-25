# ECHELON — Predictive Maintenance Analytics

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-blue.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Charts-Plotly-purple.svg?logo=plotly&logoColor=white)](https://plotly.com/)

> **Predict before failure.** An end-to-end Python analytics and predictive modeling project that forecasts industrial equipment failure risk from operating telemetry and delivers interactive decision support through a multi-page Streamlit dashboard.

---

## Overview

**ECHELON** is a data analytics and machine learning application designed for industrial condition monitoring and predictive maintenance. Built using the **AI4I 2020 Predictive Maintenance Dataset** and an **Extreme Gradient Boosting (XGBoost)** classifier, ECHELON addresses the real-world challenge of equipment downtime by identifying failure risk patterns before catastrophic breakdowns occur.

The project combines rigorous data preprocessing, cost-sensitive machine learning to handle severe class imbalance (3.39% failure incidence), and an interactive, light-themed **Streamlit** dashboard featuring live telemetry simulation, manual on-demand diagnostics, comparative what-if scenario planning, and full model transparency.

---

## Problem Statement

In industrial manufacturing, equipment maintenance traditionally follows two costly paradigms:
1. **Reactive Maintenance (Run-to-Failure):** Operating machines until failure causes unplanned downtime, emergency repair costs, collateral tooling damage, and production delays.
2. **Preventative Maintenance (Time-Based):** Servicing machinery at arbitrary calendar intervals leads to premature component replacement and labor inefficiency while still missing unexpected anomaly-driven failures.

**Predictive Maintenance (PdM)** uses sensor telemetry to monitor equipment health continuously and forecast failure likelihood. However, building effective PdM models presents distinct data analytics challenges:
* **Severe Class Imbalance:** Failures represent a tiny fraction of operational records ($\approx 3.4\%$). Naive models optimize for accuracy by predicting "no failure" across all cases, missing critical downtime events.
* **Operational Interpretability:** Technicians need actionable context (e.g., specific thresholds and component advisories), not just raw probability scores.
* **Decision Support:** Operations teams need the ability to simulate "what-if" operational adjustments (e.g., derating spindle speed or torque) to evaluate risk changes prior to execution.

---

## Objectives

* **Audit and preprocess** the 10,000-sample AI4I 2020 industrial dataset to ensure zero-loss data hygiene.
* **Engineer a robust preprocessing pipeline** that encodes product quality variants, standardizes numeric sensor readings without data leakage, and sanitizes feature schemas for gradient boosting compatibility.
* **Train a calibrated XGBoost classifier** configured with cost-sensitive class weighting (`scale_pos_weight = 28.52`) to penalize minority failure misclassifications.
* **Rigorously evaluate performance** using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix analysis on a held-out stratified test set.
* **Formulate derived diagnostic metrics**, including a standardized Machine Health Score ($0 - 100$), discrete risk tiers (*Low, Moderate, High, Critical*), and a rule-based maintenance advisory engine.
* **Deploy an interactive 4-page Streamlit analytics dashboard** supporting simulated telemetry streaming, single-instance diagnostics, comparative scenario simulation, and model interpretability.

---

## Key Features

* **Live Machine Monitoring (Simulated Telemetry Feed):** Streams real records sequentially from the AI4I 2020 dataset, featuring telemetry controls (`Start Monitoring`, `Pause`, `Next Reading`, `Prev Reading`, `Reset`) and quick-select presets for specific failure modes (e.g., Power Failure, Overstrain, Tool Wear).
* **Historical Risk Trend Line:** Interactive Plotly timeline tracking failure risk percentage across recent telemetry cycles with 40% (Warning) and 70% (Critical) threshold markers.
* **Manual Machine Diagnostics:** On-demand inference interface with input range validation, sample presets, diagnostic status indicators, and probability formatting ($<0.01\%$ for very low risks).
* **Rule-Based Maintenance Advisory Engine:** Transparent diagnostic layer mapping sensor exceedances (e.g., tool wear $\ge 200\text{ min}$, torque $\ge 60\text{ Nm}$, thermal dissipation delta $< 8.6\text{ K}$) to concrete maintenance actions.
* **What-If Comparative Simulator:** Side-by-side condition comparison (*Current vs. Simulated Conditions*) calculating absolute risk deltas ($\pm\Delta\text{ pp}$) alongside an interactive parametric sensitivity curve generator.
* **Model Insights & Transparency Portal:** Full visibility into model performance metrics, Plotly confusion matrix heatmap, ROC curve with AUC, XGBoost feature importance bar chart, and class imbalance explanations.

---

## Dataset

* **Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI Machine Learning Repository, Dataset ID: 601 / Matzka, 2020).
* **Dataset Dimensions:** 10,000 rows $\times$ 14 columns.
* **Target Variable:** `Machine failure` — Binary label ($0 = \text{Healthy / No Failure}$, $1 = \text{Machine Failure}$).
* **Class Distribution:**
  * **No Failure (Class 0):** 9,661 samples ($96.61\%$)
  * **Machine Failure (Class 1):** 339 samples ($3.39\%$)

### Feature Schema

| Column Name | Data Type | Physical Unit | Description & Operational Range |
|---|---|---|---|
| `UDI` | Integer | — | Unique record identifier ($1 - 10,000$) — *Dropped* |
| `Product ID` | String | — | Quality variant identifier (`L`/`M`/`H` + serial) — *Dropped* |
| `Type` | Categorical | — | Product quality variant: `L` (Low, 50%), `M` (Medium, 30%), `H` (High, 20%) |
| `Air temperature [K]` | Float | Kelvin (K) | Ambient room temperature ($295.3 - 304.5\text{ K}$, $\mu = 300.0\text{ K}$) |
| `Process temperature [K]` | Float | Kelvin (K) | Cutting zone process temperature ($305.7 - 313.8\text{ K}$) |
| `Rotational speed [rpm]` | Integer | rpm | Spindle rotational speed calculated from power ($1168 - 2886\text{ rpm}$) |
| `Torque [Nm]` | Float | Nm | Spindle torque output ($3.8 - 76.6\text{ Nm}$) |
| `Tool wear [min]` | Integer | Minutes | Accumulated cutting tool wear time ($0 - 253\text{ min}$) |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Binary | 0 / 1 | Specific failure mode indicators — *Dropped to prevent target leakage* |

---

## Data Preparation

The preprocessing workflow (`src/preprocessing.py`) executes the following operations:

1. **Missing Value & Duplicate Audit:** Verified complete data hygiene: 0 missing values across all columns and 0 duplicate rows.
2. **Feature Pruning:**
   * Removed arbitrary identifiers (`UDI`, `Product ID`).
   * Removed individual failure mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) which only occur when `Machine failure = 1`, preventing target leakage.
3. **Feature Header Sanitization:** Renamed square bracket headers (e.g., `Air temperature [K]`) to parenthetical formats (`Air temperature (K)`) to comply with XGBoost C++ parser requirements.
4. **Categorical Encoding:** Applied `LabelEncoder` to the 3-level product variant `Type` ($H = 0, L = 1, M = 2$).
5. **Stratified Partitioning:** Executed an 80/20 stratified split (`random_state=42`) to preserve the 3.39% minority failure proportion across both sets:
   * **Training Partition:** 8,000 samples (7,729 Healthy, 271 Failures)
   * **Testing Partition:** 2,000 samples (1,932 Healthy, 68 Failures)
6. **Feature Standardization:** Fit a `StandardScaler` strictly on the training partition and transformed both train and test sets to prevent data snooping.
7. **Artifact Persistence:** Serialized the fitted `StandardScaler`, `LabelEncoder`, and ordered feature list to `models/` using `joblib` for reproducible inference.

---

## Exploratory Data Analysis

Key findings from dataset inspection:
* **Severe Imbalance:** Only 339 failure records exist across 10,000 cycles ($1 : 28.5$ failure-to-normal ratio).
* **Temperature Correlation:** Ambient air temperature and process temperature exhibit a strong positive correlation ($\approx +0.876$), with normal operating temperature differences hovering between $8.6\text{ K}$ and $11.5\text{ K}$.
* **Torque vs. Speed Relationship:** In accordance with physical spindle power constraints ($P = \tau \cdot \omega$), rotational speed and torque are inversely related. High torque at low speed or low torque at excessive speed represents mechanical stress regimes.
* **Tool Wear Accumulation:** Tool wear increases linearly with machining cycles; tool failure incidence spikes sharply as wear surpasses 200 minutes.

---

## Machine Learning Model

### Algorithm Selection
The project exclusively utilizes **Extreme Gradient Boosting (`XGBClassifier`)** from the `xgboost` library. Gradient boosting was selected because of its proven performance on structured tabular sensor data, built-in regularization, and native handling of non-linear feature interactions.

### Hyperparameter Configuration

```python
xgb.XGBClassifier(
    n_estimators=200,          # 200 boosting iterations
    max_depth=6,               # Maximum tree depth for interaction capture
    learning_rate=0.1,         # Step size shrinkage
    scale_pos_weight=28.52,    # Cost-sensitive weighting for minority failure class
    random_state=42,           # Deterministic reproducibility
    eval_metric='logloss'      # Binary logarithmic loss objective
)
```

### Class Imbalance Handling
To prevent the model from trivially predicting "no failure" on majority class instances, `scale_pos_weight` was configured based on the training set distribution:

$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{7,729}{271} \approx 28.52$$

This assigns $28.52\times$ higher loss penalty to missed failure cases (False Negatives), driving the model to prioritize defect detection.

---

## Model Evaluation

Evaluation was performed on the held-out stratified test set ($n = 2,000$ samples, containing 68 actual failure cases). All reported metrics reflect empirical results produced by `train_model.py`:

### Performance Metrics

| Evaluation Metric | Empirical Test Score | Operational Interpretation |
|---|---|---|
| **Accuracy** | **98.15%** (0.9815) | Overall correct classification rate across test partition |
| **Precision** | **70.67%** (0.7067) | When the model alerts a failure, it is correct 7 out of 10 times |
| **Recall (Sensitivity)** | **77.94%** (0.7794) | The model successfully detects 53 out of 68 true machine failures |
| **F1-Score** | **74.13%** (0.7413) | Harmonic mean balancing precision and sensitivity under class skew |
| **ROC-AUC** | **96.72%** (0.9672) | High separability between failure and healthy probability distributions |

### Confusion Matrix ($n = 2,000$)

```
                   Predicted: Healthy (0)    Predicted: Failure (1)
Actual: Healthy (0)        1,910 (TN)                 22 (FP)
Actual: Failure (1)           15 (FN)                 53 (TP)
```

* **True Negatives ($TN$):** 1,910 — Correctly identified normal operating cycles.
* **False Positives ($FP$):** 22 — False alarms requiring inspection (manageable operational overhead).
* **False Negatives ($FN$):** 15 — Undetected failure events.
* **True Positives ($TP$):** 53 — Impending failures successfully caught before breakdown.

### Feature Importance (Split Contribution)

The XGBoost model identifies the following relative feature importance ranking:

1. **Rotational Speed ($35.01\%$):** Dominant predictor; severe speed fluctuations signal drive and power anomalies.
2. **Torque ($26.85\%$):** Strong indicator of spindle overload, mechanical resistance, and overstrain.
3. **Tool Wear ($24.14\%$):** Cumulative wear directly precipitates tool breakage failures.
4. **Air Temperature ($6.28\%$):** Sets the thermal ambient baseline.
5. **Process Temperature ($4.67\%$):** Reflects cutting zone heat dissipation.
6. **Product Quality Variant Type ($3.04\%$):** Captures material variant resistance characteristics.

> *Note: Feature importance scores reflect mathematical split contributions across tree ensembles, not physical causation.*

---

## Streamlit Application

The user interface (`app.py`) is styled with a light industrial visual design (warm ivory `#FAF8F5`, charcoal text `#1A1A1A`, and burnt orange `#C4662B` accents) organized into four distinct pages:

### 1. 📊 Overview (Live Machine Monitoring)
* **Simulated Telemetry Stream:** Ingests actual rows from `ai4i2020.csv` to emulate a live IIoT sensor feed.
* **Feed Controls:** `Start Monitoring` (auto-step timer), `Pause`, `Next Reading`, `Prev Reading`, `Reset Feed`, and a quick-select dropdown to jump directly to known anomaly records (e.g., Row #51 Power Failure, Row #70 Overstrain, Row #78 Tool Wear).
* **Health & Status Display:** Real-time Machine Health Score gauge ($0 - 100$), failure risk percentage, risk tier badge (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), and 6 sensor reading cards.
* **Risk Trend Line:** Plotly historical line chart showing risk trajectories across consecutive cycles with 40% and 70% threshold lines.
* **Architecture Flowchart:** Visual overview of the telemetry processing flow.

### 2. 🔍 Analyze Machine (Manual Diagnostics)
* **Custom Parameter Entry:** Numeric inputs and dropdowns for Product Type, Air Temp, Process Temp, Rotational Speed, Torque, and Tool Wear.
* **Sample Presets:** `Load Sample (Nominal)`, `Load Sample (High Torque)`, `Load Sample (Tool Wear)` for instant evaluation.
* **Physical Boundary Validation:** Automatic warning banners for out-of-range physical parameters.
* **Diagnostic Output:** Central diagnostic status card with failure probability (formatted as `<0.01%` for very low values), no-failure probability, health score, and risk tier.
* **Maintenance Advisories:** Deterministic rule-based advice aligned with predicted risk and sensor values.

### 3. 🔄 What-If Simulator (Scenario Comparison)
* **Dual Operating Conditions:** Side-by-side interactive sliders for *Current Operating Condition* and *Simulated Operating Condition*.
* **Risk Delta Calculation:** Displays current risk, simulated risk, and absolute risk change in percentage points ($\pm\Delta\text{ pp}$) with contextual color coding.
* **Parametric Sensitivity Curve:** Select any single operating variable to generate an interactive Plotly sensitivity sweep across 45 intervals while holding other parameters fixed, plotting Current and Simulated operating points.

### 4. 📈 Model Insights (ML Evaluation & Transparency)
* **Performance Metric Cards:** Accuracy (98.15%), Precision (70.67%), Recall (77.94%), F1-Score (74.13%), and ROC-AUC (96.72%).
* **Interactive Evaluation Tabs:**
  * *Confusion Matrix:* Interactive Plotly heatmap with test set counts.
  * *ROC Curve:* Interactive Plotly ROC plot with $\text{AUC} = 0.9672$ and random reference line.
  * *Feature Importance:* Horizontal Plotly bar chart of XGBoost feature contributions.
* **Dataset Specifications Table:** Overview of sample counts, partition sizes, and class ratios.
* **Class Imbalance Callout:** Detailed explanation of why accuracy alone is insufficient in imbalanced predictive maintenance.

---

## Tech Stack

* **Programming Language:** Python 3.13
* **Machine Learning:** `xgboost` (v3.4.1), `scikit-learn` (v1.7.1)
* **Data Processing & Analytics:** `pandas` (v3.0.2), `numpy` (v2.3.2)
* **Interactive Dashboard:** `streamlit` (v1.61.1)
* **Data Visualization:** `plotly` (v6.9.0), `matplotlib` (v3.10.8), `seaborn` (v0.13.2)
* **Model Serialization:** `joblib` (v1.5.2)

---

## Project Structure

```text
echelon/
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration (warm ivory, burnt orange)
├── data/
│   ├── ai4i2020.csv            # AI4I 2020 Predictive Maintenance Dataset (10,000 records)
│   └── README.md               # Dataset documentation, column schema, and origin
├── models/
│   ├── xgboost_model.joblib    # Trained XGBoost Classifier binary
│   ├── scaler.joblib           # Fitted StandardScaler artifact
│   ├── label_encoder.joblib    # Fitted LabelEncoder artifact for Product Type
│   ├── feature_names.joblib    # Ordered list of training feature names
│   ├── evaluation_results.json # Serialized evaluation metrics and curve coordinates
│   ├── confusion_matrix.png    # Static confusion matrix visualization
│   ├── roc_curve.png           # Static ROC curve plot
│   └── feature_importance.png  # Static feature importance bar chart
├── src/
│   ├── __init__.py             # Package initializer
│   ├── preprocessing.py        # Data loading, inspection, encoding, and scaling pipeline
│   ├── prediction.py           # Model artifact loading and single-instance prediction
│   └── evaluation.py          # Metric calculations, confusion matrix, ROC, feature plots
├── .gitignore                  # Git ignore rules for Python, cache, and virtual environments
├── app.py                      # Multi-page Streamlit analytics dashboard
├── requirements.txt            # Project dependencies
├── train_model.py              # End-to-end training and evaluation pipeline script
└── README.md                   # Comprehensive project documentation
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/echelon.git
cd echelon
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1. (Optional) Run the Training Pipeline
To inspect data distributions, fit the XGBoost model, evaluate metrics, and regenerate all model artifacts:
```bash
python train_model.py
```

### 2. Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Results / Insights

1. **Class Imbalance Resolution:** Applying `scale_pos_weight = 28.52` enabled the XGBoost classifier to capture **77.94% of true failures** (53 out of 68 in the test set) while maintaining an overall accuracy of **98.15%** and a precision of **70.67%**.
2. **Key Physical Predictors:** Rotational speed (35.0%), torque (26.9%), and tool wear (24.1%) collectively account for $>86\%$ of the model's split decisions, confirming that spindle dynamics and mechanical tool wear are the primary predictors of failure in milling equipment.
3. **High Discriminative Capability:** A test ROC-AUC score of **0.9672** demonstrates strong model capability in ranking failure probabilities across operational states.
4. **Actionable Decision Support:** Coupling calibrated probability scores with deterministic maintenance rules provides clear diagnostic recommendations (e.g., distinguishing tool wear replacement from spindle torque overload).

---

## Limitations

* **Synthetic Data Source:** The AI4I 2020 dataset is a synthetic benchmark reflecting realistic milling machine equations. It does not account for factory-floor noise, missing sensor dropouts, or uncalibrated sensor drift.
* **Tabular Snapshot Representation:** The model treats each operating cycle as an independent tabular observation rather than modeling temporal sequence dependencies across time-series windows.
* **Single Binary Failure Target:** The model predicts whether *any* failure occurs (`Machine failure = 1`), rather than directly multi-classifying specific failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) in a multi-output architecture.
* **Static Rule Layer:** Maintenance advisories are generated via predefined operational thresholds rather than dynamic reinforcement learning or historical maintenance logs.

---

## Future Improvements

* [ ] **Multi-Class Failure Mode Classification:** Train dedicated multi-output classifiers to predict specific failure types (Tool Wear, Heat Dissipation, Power Failure, Overstrain).
* [ ] **Remaining Useful Life (RUL) Prognostics:** Implement time-series regression models (e.g., LSTM, GRU, Temporal Convolutional Networks) to forecast estimated operating hours until failure.
* [ ] **Explainable AI (SHAP Integration):** Add local Shapley Additive Explanations (SHAP) waterfall plots to provide individualized feature contribution breakdowns for any manual input.
* [ ] **Live IIoT Protocol Integration:** Connect telemetry ingestion directly to MQTT or OPC-UA broker streams from industrial PLCs.


