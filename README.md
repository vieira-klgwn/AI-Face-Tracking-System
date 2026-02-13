# 🚀 Face Recognition System
### ArcFace + ONNX + 5-Point Alignment (Real-Time, CPU Only)

A clean and practical **real-time face recognition pipeline** that detects, aligns, embeds, and recognizes faces without requiring a GPU.

Built for:
- learning how face recognition works internally
- research and academic projects
- embedded or low-resource systems
- production-style experimentation

---

## ✨ Features

- Real-time face detection
- 5-point facial landmark alignment (112×112 crops)
- ArcFace 512‑D embeddings
- Cosine similarity matching
- Multi-person enrollment
- Persistent face tracking
- Threshold auto-evaluation
- CPU-only inference
- Modular and easy-to-extend design

---

## 🧠 Pipeline Overview

```
Camera
   ↓
Face Detection
   ↓
Landmarks (5 points)
   ↓
Alignment (112x112)
   ↓
ArcFace Embedding (512D)
   ↓
Cosine Similarity
   ↓
Recognized / Unknown
```

Simple idea:
> Image → Align → Convert to numbers → Compare → Decide identity

---

## 📁 Project Structure

```
FaceRecognition/
├── src/
│   ├── camera.py
│   ├── detect.py
│   ├── landmarks.py
│   ├── align.py
│   ├── embed.py
│   ├── enroll.py
│   ├── evaluate.py
│   └── recognise.py
├── data/
├── models/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone

```bash
git clone <your-repo-url>
cd FaceRecognition
```

### 2. Create virtual environment

Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Required Models

### ArcFace ONNX model

Place your ArcFace ONNX file here:

```
models/embedder_arcface.onnx
```

Expected:
- Input: 112×112
- Output: 512‑dim embedding

---

### MediaPipe landmarker

Place:

```
face_landmarker.task
```

in the project root.

---

## ▶️ Quick Start

Test step-by-step:

```bash
python -m src.camera
python -m src.detect
python -m src.align
python -m src.embed
```

Enroll people:

```bash
python -m src.enroll
```

Find best threshold:

```bash
python -m src.evaluate
```

Run live recognition:

```bash
python -m src.recognise
```

---

## 🎮 Controls (Live Mode)

| Key | Action |
|-----|---------|
| q | quit |
| r | reload database |
| + / - | adjust threshold |
| d | debug overlay |
| t | toggle tracking |

---

## 🎯 Enrollment Tips

For best accuracy:

- capture 15–20 samples per person
- vary pose and expression
- use good lighting
- keep faces centered
- avoid blur

---

## 📏 Threshold Concept

Embeddings are compared using cosine distance.

- lower distance → same person
- higher distance → different person

Always run `evaluate.py` to automatically compute the best threshold for your data.

Typical range: **0.30 – 0.40**

---

## 💡 Why Embeddings Instead of Images?

- smaller storage
- faster comparisons
- better generalization
- scalable to many identities

We compare **numbers**, not raw pixels.

---

## 🖥 System Requirements

- CPU only (no GPU needed)
- 2GB+ RAM recommended
- webcam
- Windows / Linux / macOS
- Python 3.8+

---

## 🛠 Troubleshooting

Camera not opening?
→ try another camera index

Poor accuracy?
→ collect more samples + evaluate threshold

Model missing?
→ verify file paths

Import errors?
→ reinstall dependencies

---

## 🎓 Learning Goals

This project helps you understand:

- face detection
- facial alignment
- feature embeddings
- similarity matching
- real-time tracking

It focuses on **understanding the pipeline**, not just calling an API.

---

## 📜 License

Educational and research use.