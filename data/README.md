# AI4I 2020 Predictive Maintenance Dataset

## Overview
This directory contains the dataset used for training and simulating the **ECHELON** predictive maintenance pipeline.

- **File**: `ai4i2020.csv`
- **Total Records**: 10,000 samples
- **Total Features**: 14 columns
- **Source**: UCI Machine Learning Repository (Dataset ID: 601) / Matzka (2020)
- **DOI**: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)

## Column Definitions

| Column | Data Type | Description |
|---|---|---|
| `UDI` | Integer | Unique Row Identifier (1 to 10000) — *Dropped during preprocessing* |
| `Product ID` | String | Variant letter (`L`/`M`/`H`) and serial number — *Dropped during preprocessing* |
| `Type` | Categorical (`L`, `M`, `H`) | Product quality variant: Low (50%), Medium (30%), High (20%) |
| `Air temperature [K]` | Float | Ambient room temperature in Kelvin (295.3 – 304.5 K) |
| `Process temperature [K]` | Float | Process temperature in Kelvin generated during milling (305.7 – 313.8 K) |
| `Rotational speed [rpm]` | Integer | Spindle rotational speed calculated from power (1168 – 2886 rpm) |
| `Torque [Nm]` | Float | Torque output normally distributed around 40 Nm (3.8 – 76.6 Nm) |
| `Tool wear [min]` | Integer | Cutting tool wear time in minutes (0 – 253 min) |
| `Machine failure` | Binary (`0`, `1`) | **Target Variable**: 0 = Healthy (9,661 samples), 1 = Failure (339 samples) |
| `TWF` | Binary | Tool Wear Failure mode indicator — *Dropped during training to prevent target leakage* |
| `HDF` | Binary | Heat Dissipation Failure mode indicator — *Dropped during training* |
| `PWF` | Binary | Power Failure mode indicator — *Dropped during training* |
| `OSF` | Binary | Overstrain Failure mode indicator — *Dropped during training* |
| `RNF` | Binary | Random Failure mode indicator — *Dropped during training* |

## Usage in ECHELON
1. **Training Pipeline (`train_model.py`)**: Uses the 6 core predictive features (`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`) to train the cost-sensitive XGBoost classifier.
2. **Live Monitoring Page (`app.py`)**: Streams actual rows from this CSV sequentially as a simulated industrial sensor feed.
