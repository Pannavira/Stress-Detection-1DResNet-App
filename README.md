# Stress Detection: Edge 1DResNet Prototype

A high-performance Android application for real-time stress detection using a **Knowledge Distilled (KD) ResNet Student Model**. This application is designed to run locally on edge devices, processing physiological signals (ECG and EDA) with high efficiency and privacy.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### 📋 Prerequisites

Before you begin, ensure you have the following installed:

*   **Node.js**: Version 18 or later ([Download](https://nodejs.org/))
*   **npm**: Usually comes with Node.js
*   **Java Development Kit (JDK)**: Version 17 ([Recommended: Zulu JDK 17](https://www.azul.com/downloads/?version=java-17-lts&package=jdk))
*   **Android Studio**: Latest version with SDK 34+ installed.
*   **Expo CLI**: Global installation is optional as it's typically run via `npx`.

---

### 🛠 Installation & Setup

#### 1. Clone the Repository
Open your terminal and run the following commands:
```bash
git clone https://github.com/Pannavira/Stress-Detection-1DResNet-App.git
cd Stress-Detection-1DResNet-App
```

#### 2. Install Dependencies
Install the required Node.js packages:
```bash
npm install
```

#### 3. Environment Configuration
Ensure your system environment variables are correctly set:
*   `ANDROID_HOME`: Path to your Android SDK (e.g., `C:\Users\Name\AppData\Local\Android\Sdk`)
*   `JAVA_HOME`: Path to your JDK 17 installation.

---

### 📱 Running the Application

Since this app uses native C++ code via `onnxruntime-react-native`, it **cannot** run in the standard Expo Go app. You must build a development client or run the native build command.

#### Run on Android Emulator/Device
1.  Connect your Android device via USB (with Debugging enabled) or start an Android Emulator.
2.  Execute the build command:
    ```bash
    npx expo run:android
    ```
    *This command will prebuild the native Android project and install the app on your device.*

#### Run on iOS (macOS only)
1.  Install CocoaPods: `cd ios && pod install && cd ..`
2.  Execute the build command:
    ```bash
    npx expo run:ios
    ```

---

## 📖 Usage Guide

1.  **Load Data**:
    *   Click the **"PICK JSON FILE"** button.
    *   You can use the provided sample file located in `assets/dummy_data.json` or any compatible WESAD-formatted JSON.
2.  **Visualize**: The top dashboard will render the ECG and EDA waveforms to verify signal integrity.
3.  **Analyze**: Press **"ANALYZE"** to trigger the ONNX inference engine.
    *   **NORMAL**: Indicates a calm/resting state.
    *   **STRESS**: Indicates detected sympathetic activation.
4.  **Monitor**: View the **Inference Latency** and **Execution Logs** at the bottom to see real-time performance metrics.

---

## 🔬 Technical Architecture

### Model Specifications
*   **Architecture**: Lightweight 1D-ResNet (Student Model).
*   **Input**: 60-second window (7680 samples) of ECG and EDA signals.
*   **Engine**: Powered by `onnxruntime-react-native` for on-device inference.
*   **Calibration**: Implements **Temperature Scaling** to provide reliable confidence scores.

### Key Technologies
*   **React Native / Expo**: Cross-platform framework.
*   **ONNX Runtime**: Cross-platform model acceleration.
*   **React Native Chart Kit**: Real-time signal visualization.

---

## 📂 Project Structure
*   `App.js`: Core logic for UI and ONNX model inference.
*   `utils/preprocessing.js`: Signal normalization and windowing logic.
*   `assets/model.onnx`: The serialized ResNet student model.
*   `assets/dummy_data.json`: Sample data for testing.

---

## 🤝 Research & Credits
**Author**: Pannavira  
**Research Area**: Knowledge Distillation for Physiological Signal Processing  
**Dataset**: Based on the WESAD (Wearable Stress and Affect Detection) dataset.

---

**Link to Repository**: [https://github.com/Pannavira/Stress-Detection-1DResNet-App](https://github.com/Pannavira/Stress-Detection-1DResNet-App)
