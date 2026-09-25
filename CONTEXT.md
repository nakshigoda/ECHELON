# ECHELON — Project Context & Developer Reference

This document serves as the technical reference manual and architecture context for the **ECHELON** predictive maintenance analytics codebase.

---

## 1. Project Purpose & Positioning

**ECHELON** is an applied Python machine learning and data analytics project developed for industrial predictive maintenance (PdM). The system predicts whether an industrial milling machine will experience a failure (`Machine failure = 1`) based on real-time operating conditions and product quality variants.

### Core Value Proposition
* **Mitigate Costly Downtime:** Predict mechanical failure risk before breakdowns halt manufacturing lines.
* **Bridge ML and Operations:** Convert statistical failure probabilities into calibrated Machine Health Scores ($0 - 100$) and deterministic, actionable maintenance advisories.
* **Interactive What-If Simulation:** Enable maintenance engineers and operators to simulate parameter adjustments (e.g., derating spindle speed or torque) and observe non-linear failure risk changes.

---

## 2. Current Implementation Status

| Component | Status | Details |
|---|---|---|
| **Data Ingestion & Cleaning** | Completed | Full audit of AI4I 2020 dataset (10,000 samples, 0 missing, 0 duplicates). |
| **Preprocessing Pipeline** | Completed | Bracket sanitization, `LabelEncoder` for `Type`, stratified 80/20 train/test split, `StandardScaler` normalization. |
| **XGBoost Classifier** | Completed | Imbalance-weighted (`scale_pos_weight = 28.52`), tuned hyperparameters, serialized artifacts (`.joblib`). |
| **Evaluation Suite** | Completed | Accuracy (98.15%), Precision (70.67%), Recall (77.94%), F1 (74.13%), ROC-AUC (96.72%), Confusion Matrix ($n=2,000$). |
| **Streamlit Dashboard** | Completed | 4 distinct pages (Live Monitoring, Manual Diagnostics, What-If Simulator, Model Insights) with custom light industrial CSS theme. |
| **Advisory Engine** | Completed | Rule-based maintenance diagnostic layer mapping multi-sensor thresholds to actionable technician steps. |

---

## 3. Dataset Details & Schema

* **Dataset Name:** AI4I 2020 Predictive Maintenance Dataset
* **Source:** UCI Machine Learning Repository (Dataset ID: 601) / Stephan Matzka (2020)
* **File Location:** `data/ai4i2020.csv` (Size: $\approx 522\text{ KB}$, 10,000 rows $\times$ 14 columns)
* **Target Label:** `Machine failure` ($0 = \text{Healthy}$, $1 = \text{Failure}$)
* **Distribution:**
  * Class 0 (No Failure): 9,661 samples ($96.61\%$)
  * Class 1 (Failure): 339 samples ($3.39\%$)

### Feature Breakdown

| Feature | Raw Header | Sanitized Header | Data Type | Physical Range | Role in Pipeline |
|---|---|---|---|---|---|
| `UDI` | `UDI` | — | Integer | $1 - 10,000$ | Identifier — **Dropped** |
| `Product ID` | `Product ID` | — | String | `L47181`, etc. | Serial identifier — **Dropped** |
| `Type` | `Type` | `Type` | Categorical | `L` (50%), `M` (30%), `H` (20%) | Product quality variant — **Label Encoded** ($H=0, L=1, M=2$) |
| `Air temp` | `Air temperature [K]` | `Air temperature (K)` | Float | $295.3 - 304.5\text{ K}$ | Ambient temperature — **Standard Scaled** |
| `Process temp` | `Process temperature [K]` | `Process temperature (K)` | Float | $305.7 - 313.8\text{ K}$ | Cutting process temperature — **Standard Scaled** |
| `Speed` | `Rotational speed [rpm]` | `Rotational speed (rpm)` | Integer | $1168 - 2886\text{ rpm}$ | Spindle angular velocity — **Standard Scaled** |
| `Torque` | `Torque [Nm]` | `Torque (Nm)` | Float | $3.8 - 76.6\text{ Nm}$ | Spindle torque — **Standard Scaled** |
| `Tool wear` | `Tool wear [min]` | `Tool wear (min)` | Integer | $0 - 253\text{ min}$ | Cumulative tool wear — **Standard Scaled** |
| `Failure modes` | `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | — | Binary | $0 / 1$ | Specific defect indicators — **Dropped** (prevents target leakage) |

---

## 4. Preprocessing & Feature Engineering Pipeline

The preprocessing logic is implemented in `src/preprocessing.py`:

```
Raw CSV [10,000 x 14]
   │
   ├── 1. Drop non-predictive columns: ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
   │
   ├── 2. Header Sanitization: Replace '[' with '(' and ']' with ')'
   │      (Avoids C++ parser errors in XGBoost: 'Air temperature [K]' -> 'Air temperature (K)')
   │
   ├── 3. Categorical Encoding: LabelEncoder fit on 'Type' (H=0, L=1, M=2) -> models/label_encoder.joblib
   │
   ├── 4. Feature Selection: X = ['Type', 'Air temperature (K)', 'Process temperature (K)',
   │                              'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)']
   │                          y = 'Machine failure'
   │
   ├── 5. Stratified Partitioning: 80% Train (8,000 samples, 271 failures)
   │                               20% Test (2,000 samples, 68 failures)
   │
   └── 6. Feature Normalization: StandardScaler fit on X_train only -> models/scaler.joblib
          Transform both X_train and X_test -> X_train_scaled, X_test_scaled
```

### Preprocessing Consistency for Inference
The function `preprocess_for_prediction` (and `src/prediction.py`) reloads the exact `label_encoder.joblib`, `scaler.joblib`, and `feature_names.joblib` artifacts to ensure identical transformation logic during runtime inference.

---

## 5. Model Architecture & Training Configuration

### Algorithm Specification
* **Classifier:** `xgboost.XGBClassifier`
* **Objective:** `binary:logistic` (`eval_metric='logloss'`)
* **Tree Parameters:** `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`
* **Reproducibility:** `random_state=42`
* **Cost-Sensitive Weighting:** `scale_pos_weight = 28.52`

### Why `scale_pos_weight = 28.52`?
In standard binary cross-entropy, all classification errors receive equal penalty. In a dataset where 96.61% of samples are non-failures, a trivial model predicting $\hat{y} = 0$ achieves 96.61% accuracy while catching zero failures.

By setting:
$$\text{scale\_pos\_weight} = \frac{\text{Count}(\text{Class } 0)}{\text{Count}(\text{Class } 1)} = \frac{7,729}{271} \approx 28.52$$
the loss contribution of False Negatives (missed machine failures) is scaled by $28.52\times$, compelling the decision trees to identify the subtle boundary conditions where failures occur.

---

## 6. Empirical Evaluation Results

Evaluated on the held-out test partition ($n = 2,000$ samples, containing 68 actual failures):

| Metric | Score | Formula / Notes |
|---|---|---|
| **Accuracy** | **98.15%** | $\frac{1910 + 53}{2000} = \frac{1963}{2000}$ |
| **Precision** | **70.67%** | $\frac{53}{53 + 22} = \frac{53}{75}$ |
| **Recall (Sensitivity)** | **77.94%** | $\frac{53}{53 + 15} = \frac{53}{68}$ (Catches 53 of 68 true failures) |
| **F1-Score** | **74.13%** | $2 \cdot \frac{0.7067 \cdot 0.7794}{0.7067 + 0.7794}$ |
| **ROC-AUC** | **96.72%** | Area under the Receiver Operating Characteristic Curve |

### Test Confusion Matrix
* **True Negatives ($TN$):** 1,910
* **False Positives ($FP$):** 22
* **False Negatives ($FN$):** 15
* **True Positives ($TP$):** 53

### XGBoost Feature Importance Breakdown
* **`Rotational speed (rpm)`:** $35.01\%$ (Dominant predictor of power and mechanical anomalies)
* **`Torque (Nm)`:** $26.85\%$ (Spindle resistance and mechanical overload)
* **`Tool wear (min)`:** $24.14\%$ (Cumulative tool wear cutting time)
* **`Air temperature (K)`:** $6.28\%$ (Ambient thermal baseline)
* **`Process temperature (K)`:** $4.67\%$ (Machining cutting zone heat dissipation)
* **`Type`:** $3.04\%$ (Material variant hardness and quality grading)

---

## 7. Streamlit Dashboard Architecture (`app.py`)

### Theme Configuration (`.streamlit/config.toml` + Custom CSS)
* **Primary Background:** `#FAF8F5` (Warm Ivory / Off-White)
* **Card & Container Background:** `#FFFFFF` with `1px solid #E5E0DA` border
* **Primary Accent:** `#C4662B` (Muted Burnt Orange)
* **Typography:** `Inter`, sans-serif (charcoal text `#1A1A1A`, secondary text `#5F5F5F`)
* **Status Badges:** Green (`#2D7D46` / `#EBF5EE`), Amber (`#C4880F` / `#FDF4E4`), Red (`#B83A2E` / `#FCEAE8`)

### Page-by-Page Breakdown

#### 1. 📊 Overview (Live Machine Monitoring)
* Emulates an industrial telemetry stream using sequential rows from `data/ai4i2020.csv`.
* Telemetry controls: `Start Monitoring` (auto-step timer with `time.sleep(1.2)`), `Pause`, `Next Reading`, `Prev Reading`, `Reset Feed`.
* Quick Select Anomaly dropdown to jump directly to specific failure records (e.g., Row 50 Power Failure, Row 69 Overstrain, Row 77 Tool Wear).
* Displays Machine ID (`Product ID`, `UDI`), Health Gauge ($0 - 100$), Failure Risk %, Risk Level, and 6 Sensor Cards.
* Interactive Plotly stream history chart displaying failure risk percentage over recent cycles with 40% and 70% threshold lines.

#### 2. 🔍 Analyze Machine (Manual Input & Diagnostics)
* Interactive numeric inputs for manual machine condition entry.
* Quick preset loaders: `Nominal Sample`, `High Torque Sample`, `Tool Wear Sample`.
* Input validation warnings against empirical physical boundaries.
* Outputs: Status Badge (`Healthy` / `Failure Predicted`), Failure Probability ($<0.01\%$ formatting for negligible risks), No-Failure Probability, Health Score, and Risk Tier.
* Transparent Rule-Based Maintenance Advisory layer.

#### 3. 🔄 What-If Simulator (Scenario Comparison)
* Dual-column layout comparing *Current Operating Condition* against *Simulated Operating Condition*.
* Computes Absolute Risk Delta ($\pm\Delta\text{ pp}$) with contextual color coding.
* Parametric Sensitivity Curve: Sweeps any chosen variable across 45 intervals, generating an interactive Plotly risk trajectory with Current and Simulated points marked.

#### 4. 📈 Model Insights (ML Evaluation & Transparency)
* 5 core performance metric cards (Accuracy, Precision, Recall, F1, ROC-AUC).
* Tabbed Plotly charts:
  * Tab 1: Confusion Matrix heatmap ($n=2,000$).
  * Tab 2: ROC Curve with AUC score and random guessing baseline.
  * Tab 3: XGBoost Feature Importance horizontal bar chart.
* Model specifications table and class imbalance explanation callout.

---

## 8. Source Code Structure & Key Modules

```text
echelon/
├── .streamlit/
│   └── config.toml          # Streamlit theme definition
├── data/
│   ├── ai4i2020.csv         # AI4I 2020 dataset (10,000 records)
│   └── README.md            # Dataset schema and source documentation
├── models/
│   ├── xgboost_model.joblib # Serialized XGBoost model
│   ├── scaler.joblib        # Serialized StandardScaler
│   ├── label_encoder.joblib # Serialized LabelEncoder
│   ├── feature_names.joblib # Serialized feature name list
│   └── evaluation_results.json # Metrics, CM, ROC points, and feature weights
├── src/
│   ├── __init__.py          # Python package initializer
│   ├── preprocessing.py     # Data cleaning, encoding, scaling, and train/test splitting
│   ├── prediction.py        # Artifact loading and single-instance inference logic
│   └── evaluation.py       # Static plotting and metric calculation functions
├── .gitignore               # Git ignore rules
├── app.py                   # Multi-page Streamlit application
├── requirements.txt         # Project dependencies
├── train_model.py           # Training pipeline script
├── README.md                # Public GitHub documentation
└── CONTEXT.md               # Technical developer reference (this document)
```

---

## 9. Design Decisions & Rationale

1. **Why XGBoost over Other Algorithms?**
   * Tabular sensor data with non-linear thresholds (e.g., torque $\times$ speed power curves) is ideally suited for gradient-boosted decision trees.
   * Native support for `scale_pos_weight` avoids synthetic oversampling distortions (such as SMOTE) which can create unrealistic sensor readings.
2. **Why Header Sanitization (`[` $\to$ `(`)?**
   * XGBoost 3.x rejects feature names containing square brackets `[` or `]`. Sanitizing column names globally across training and prediction prevents runtime C++ parsing failures.
3. **Why a Light Industrial Theme?**
   * Avoids the generic dark-blue corporate template or neon gradients, matching the visual aesthetic of modern industrial control room software.
4. **Why Rule-Based Advisories Separate from ML?**
   * Clearly separates probabilistic predictions from deterministic operational guidance, ensuring operators understand why specific maintenance tasks (e.g., tool replacement) are recommended.

---

## 10. Current Limitations & Known Considerations

* **Synthetic Data:** The AI4I 2020 dataset is a mathematical simulation; it does not capture industrial sensor noise, missing packets, or sensor calibration drift.
* **Single Snapshot Predictions:** Telemetry cycles are treated as independent tabular rows rather than sequential temporal time windows.
* **Binary Target Only:** Does not directly multi-classify failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) in a multi-output configuration.

---

## 11. How to Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/<username>/echelon.git
cd echelon

# 2. Set up virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Re-train the model
python train_model.py

# 5. Launch the Streamlit dashboard
python -m streamlit run app.py
```
App runs at `http://localhost:8501`.
