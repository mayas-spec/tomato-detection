# TomatoVision AI 🍅

An AI-powered tomato plant disease detection platform built as a Computer Vision Capstone Project. The app uses a two-stage deep learning pipeline to identify tomato leaves and classify diseases in real time.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tomato-detection.streamlit.app)

---

## Features

- **Two-stage AI pipeline** — Gate model filters non-tomato images; disease classifier identifies the condition
- **3 disease classes** — Healthy, Early Blight, Late Blight
- **Explainable AI (XAI)** — Occlusion sensitivity heatmap shows which leaf regions drove the prediction
- **Disease library** — Searchable reference for symptoms, treatment, and prevention
- **Model performance analytics** — Training curves, confusion matrices, ROC curves
- **Prediction history** — Session log with CSV export
- **Mobile-responsive** — Works on phones and tablets

---

## Model Architecture

Both models use **EfficientNetB0** fine-tuned on the PlantVillage dataset and exported to ONNX for cross-platform inference.

| Model | Task | Accuracy |
|---|---|---|
| Gate model (`gate_phase2.onnx`) | Tomato leaf vs. everything else | 99.93% |
| Disease classifier (`detect_phase2.onnx`) | Healthy / Early Blight / Late Blight | 96.16% |

> **Note:** Models were trained on leaf images from the PlantVillage dataset. Upload clear photos of tomato **leaves** (not fruit) for best results.

---

## Running Locally

```bash
# 1. Create environment
conda create -n tomato python=3.11
conda activate tomato

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## Tech Stack

- **Frontend** — Streamlit, Plotly, custom glassmorphism CSS
- **Inference** — ONNX Runtime (no TensorFlow required at runtime)
- **Models** — EfficientNetB0 trained with Keras/TensorFlow, exported via tf2onnx
- **XAI** — Occlusion sensitivity (gradient-free, ONNX-compatible)

---

## Project Structure

```
tomato-detection/
├── app.py                          # Streamlit app (all 6 pages)
├── models/
│   ├── gate_phase2.onnx            # Stage 1: tomato leaf gate
│   └── detect_phase2.onnx          # Stage 2: disease classifier
├── requirements.txt
└── *.ipynb                         # Training notebooks
```
