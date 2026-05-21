# ⚽ Football Image Understanding: From Bounding-Box Crops Classification to End-to-End Object Detection (YOLOv8)

A comprehensive deep learning experimentation framework designed to analyze and understand football match imagery. This project transitions from a modular academic baseline (evaluating MLPs, CNNs, RNNs, and LSTMs on cropped regions) to a fully deployed end-to-end real-time object detection system using **YOLOv8** and a **Gradio** web interface.

---

## 📌 Project Overview & Objectives

The goal of this project is to develop a robust pipeline for detecting and classifying key elements in professional football matches (Players, Referees, Goalkeepers, and Balls). 

The repository is structured into two main phases:
1. **Academic Baseline Phase:** Rather than testing complex detection architectures from scratch, we isolate localized bounding boxes into **64x64 pixel crops** to evaluate and benchmark raw sequence and spatial feature extraction capabilities across standard architectures (MLP, CNN, RNN, LSTM).
2. **End-to-End Deployment Phase:** Leveraging the insights from the baseline phase, we implement and train a full **YOLOv8** object detection model, deploying it into an interactive user-facing application.

---

## 📊 Dataset Pipeline & Structure

The dataset is securely pulled from **Roboflow** (Project: `testing_football`, Workspace: `mohamed-houij`).

### Data Splits & Volume
* **Train Set:** 12,986 annotations across 5,934 unique images
* **Validation Set:** 1,478 annotations across 602 unique images
* **Test Set:** 371 annotations across 149 unique images

### Target Classes (11 total)
The framework maps and standardizes 11 target categories present in the dataset:
`Arbitre`, `Ballon`, `Football`, `Gardien`, `Joueur`, `Player`, `ball`, `big`, `person`, `player`, `refere`.

### Preprocessing Blueprint (`safe_crop`)
For the benchmarking phase, a custom preprocessing pipeline was developed to extract safe bounding box regions:
* Bounding limits (`xmin, ymin, xmax, ymax`) are read and verified.
* Images are safely cropped, normalized to a `float32` range between `0.0` and `1.0`, and dynamically resized to a uniform **64x64x3** matrix.

---

## 🔬 Phase 1: Architectural Experiments (Baseline Classification)

We sequentially engineered, hyper-tuned, and benchmarked several architectures on the object crops:

* **Multi-Layer Perceptrons (MLPs):** Established a pixel-level dense baseline.
* **Hyperparameter & Regularization Tuning:** Systematically evaluated standard activation functions (ReLU vs variants), adaptive learning rates, and anti-overfitting components (Dropout layers and $L_2$ weight decay).
* **Convolutional Neural Networks (CNNs):** Implemented spatial feature extractors to properly capture local shapes, player jersey details, and context.
* **Recurrent Architectures (Vanilla RNN & LSTM):** Experimented with sequence-based classification by processing image row/patch matrices as sequence timestamps to test spatial sequence dependencies.

---

## 🚀 Phase 2: Object Detection (YOLOv8) & Results

The final stage of the project introduces a unified **YOLOv8 Nano (YOLOv8n)** architecture trained for 50 epochs on the native image dataset. It achieves competitive localization metrics, handling simultaneous multi-class object detection.

### 📈 Training & Validation Performance
The model successfully converged over the 50-epoch cycle:

* **Validation Set:** **$42.79\%$ mAP50**
* **Test Set:** **$66.18\%$ mAP50**
