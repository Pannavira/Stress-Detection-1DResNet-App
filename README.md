# Stress Detection: Edge Detection Prototype

A research-grade Android application for real-time stress detection using a **Knowledge Distilled (KD) ResNet Student Model**. This application is designed to run locally on edge devices, processing physiological signals (ECG and EDA) from the WESAD dataset with high efficiency and accuracy.

---

## 🚀 Key Features

- **Local Inference Engine**: Powered by `onnxruntime-react-native` for low-latency, private, and offline stress classification.
- **Signal Visualization**: Real-time rendering of ECG (Heart Activity) and EDA (Skin Conductance) waves using `react-native-chart-kit`.
- **Research Data Support**: Integrated file picker to load and analyze custom WESAD signal slices in JSON format.
- **Scientific Calibration**: Implements **Post-Hoc Temperature Scaling** to align model confidence with empirical research results.
- **Performance Tracking**: Live display of inference latency (ms) and raw model logits.

---

## 🛠 Technical Architecture

### Model Specifications
- **Architecture**: Lightweight ResNet-based Student Model.
- **Input Shape**: `(1, 2, 7680)` — 1 Batch, 2 Channels (ECG, EDA), 60-second window.
- **Sampling Rate**: 140Hz (derived from 700Hz original via decimation factor 5).
- **Complexity**: ~120k Parameters.
- **Format**: Mobile-optimized ONNX.

### Signal Pipeline
1. **Filtering**: Pre-filtered using 4th-order Butterworth bandpass (ECG: 0.5-40Hz, EDA: 0.05-5Hz).
2. **Normalization**: Global subject-wise Z-score normalization to preserve physiological variance.
3. **Segmentation**: 60-second non-overlapping windows (7680 samples).

---

## 📥 Installation

### Prerequisites
- **Node.js** (v18+) & **npm**
- **Android Studio** with SDK 34+
- **Java JDK 17** (Ensure `JAVA_HOME` is set)
- **Expo CLI** (`npm install -g expo-cli`)

### Setup Steps
1. **Clone and Install Dependencies**:
   ```bash
   cd aplikasi
   npm install
   ```

2. **Configure Environment**:
   Ensure your system environment variables include:
   - `ANDROID_HOME`: Path to your Android Sdk folder.
   - `JAVA_HOME`: Path to your Android Studio JBR or OpenJDK folder.

3. **Build the Development Client**:
   This app uses native C++ code (ONNX), so it requires a custom build (not Expo Go):
   ```bash
   npx expo run:android
   ```

---

## 📖 Usage Guide

1. **Prepare Data**: Use the provided Python script `wesad_to_json.py` to generate research-grade JSON slices from the WESAD `.pkl` files.
2. **Load Data**:
   - Transfer the `.json` files to your Android device/emulator (Downloads folder).
   - Click **"PICK JSON FILE"** in the app and select your sample.
3. **Visualize**: View the physiological signal patterns in the top dashboard to verify data integrity.
4. **Analyze**: Press **"RUN RESEARCH ANALYSIS"**.
   - **NORMAL (Green)**: Indicates parasympathetic dominance/resting state.
   - **STRESS (Red)**: Indicates sympathetic activation/high arousal.

---

## 🔬 Research Note: Model Calibration

During deployment, we observed that models trained via Knowledge Distillation (KD) with high temperature targets ($T=4.0$) produce conservative probabilities. 

To ensure the user interface provides interpretable confidence scores without altering the underlying classification logic, we applied **Temperature Scaling** ($T=0.125$) during the post-processing phase. This aligns the visual certainty (80-95%) with the model's objective accuracy (82%+) as reported in the associated research paper.

---

## 📂 Project Structure
- `App.js`: Main UI and ONNX inference logic.
- `utils/preprocessing.js`: Mathematical implementation of Z-score normalization and signal trimming.
- `assets/model.onnx`: The serialized ResNet student model.
- `android/`: Native Android project configurations (Legacy Architecture enabled for ONNX compatibility).

---

**Author**: Pannavira  
**Research Area**: Knowledge Distillation for Physiological Signal Processing  
**Dataset**: WESAD (Wearable Stress and Affect Detection)
