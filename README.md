# 🔍 Time Series Projects with LSTM: Stock Prediction & Anomaly Detection (PyTorch)

This repository contains three time-series projects using **LSTM networks in PyTorch**, demonstrating use cases in **stock price prediction** and **sensor anomaly detection**. The final project also includes a **manually implemented LSTM**, showcasing how LSTM gates function under the hood.

---

## 📁 Project Files

### 1. `apple_stock_lstm.ipynb` 📈
- **Objective**: Predict Apple's stock closing prices using past 30-day sequences.
- **Framework**: PyTorch
- **Approach**:
  - Data preprocessing with `MinMaxScaler`
  - Sequence generation for LSTM
  - One-step-ahead stock price forecasting
  
<p align="center">
  <img src="https://raw.githubusercontent.com/Mayankgbrc/RNN-LSTM/refs/heads/main/images/output_apple_share.jpg" align="center" width="90%" alt="Grayscale Image" />
 </p>
---

### 2. `anomaly_detection_lstm_pytorch.ipynb` ⚠️
- **Objective**: Detect **harsh anomalies** in synthetic sensor data.
- **Framework**: PyTorch
- **Approach**:
  - Train LSTM on clean (normal) sensor sequences
  - Use prediction error to flag anomalies
  - Threshold set using error percentiles


<p align="center">
  <img src="https://raw.githubusercontent.com/Mayankgbrc/RNN-LSTM/refs/heads/main/images/output_harsh_anomaly.jpg" align="center" width="90%" alt="Grayscale Image" />
 </p>
---

### 3. `anomaly_detection_custom_lstm.ipynb` 🧠
- **Objective**: Detect **mild/subtle anomalies** using both standard PyTorch LSTM and a **manually implemented LSTM**.
- **Framework**: PyTorch
- **Features**:
  - Custom synthetic sensor data generation with subtle pattern breaks
  - Standard PyTorch LSTM-based detection
  - **Manual LSTM cell implementation** (forward pass with explicit gate logic)
  - Comparison between framework and hand-crafted LSTM behavior

<p align="center">
  <img src="https://raw.githubusercontent.com/Mayankgbrc/RNN-LSTM/refs/heads/main/images/output_small_anomaly.jpg" align="center" width="90%" alt="Grayscale Image" />
  </p>
---

## 🧪 Use Cases

| File | Model | Anomaly Type | Implementation |
|------|--------|---------------|------------------|
| `apple_stock_lstm.py` | PyTorch LSTM | None (forecasting only) | PyTorch |
| `anomaly_detection_lstm_pytorch.py` | PyTorch LSTM | Harsh & obvious | PyTorch |
| `anomaly_detection_custom_lstm.py` | PyTorch + Manual LSTM | Subtle drops | PyTorch + Raw NumPy |

---

## 🚀 Getting Started

Clone this repo:
   ```bash
   git clone https://github.com/Mayankgbrc/RNN-LSTM.git
   cd RNN-LSTM
   ```
