# 🚗 Smart Parking ALPR System

An end-to-end Automatic License Plate Recognition (ALPR) system for smart parking management, built for **42028 Deep Learning and CNN — Assignment 3** at the University of Technology Sydney.

---

## 🎯 What it does

Upload an image, video, or use your webcam — the system automatically:
- Detects license plates using **YOLOv26s** 
- Enhances dark/night images using **Zero-DCE++**
- Reads plate text using fine-tuned **PARSEQ** (supports both Latin and Chinese plates)
- Tracks vehicles across video frames using **ByteTrack**
- Logs parking check-in / check-out events with fee calculation
- Displays all sessions on a live dashboard

---

## 🏗️ Pipeline

```
Input (Image / Video / Webcam)
        ↓
Stage 1: Zero-DCE++ (dark images only)
        ↓
Stage 2: YOLOv26s — plate detection
        ↓
Stage 3: Crop + Resize (128×32 px)
        ↓
Stage 4: PARSEQ-FT — OCR (Latin + Chinese)
        ↓
Stage 5: ByteTrack + Majority Vote (video only)
        ↓
Stage 6: Parking Session Events → SQLite
        ↓
Streamlit Dashboard
```

---

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| YOLOv26s | mAP@0.5 | 0.9939 |
| YOLOv26s | Precision | 0.9947 |
| YOLOv26s | Recall | 0.9901 |
| PARSEQ-FT | Latin plate accuracy | 0.78 |
| PARSEQ-FT | Chinese plate accuracy | 0.79 |
| ByteTrack + vote | Temporal lift | +6.7% |

### Per-Condition OCR Accuracy

| Condition | Accuracy |
|-----------|----------|
| Normal | 100% |
| Weather/Rain | 98.5% |
| Rotate | 93.5% |
| Tilt | 72% |
| Blur | 67.5% |
| Night | 50.5% |

---

## 🛠️ Tech Stack

- **Detection**: YOLOv26s (Ultralytics)
- **OCR**: PARSEQ (fine-tuned, 70-char vocab: Latin + Chinese)
- **Night Enhancement**: Zero-DCE++
- **Tracking**: ByteTrack
- **Web App**: Streamlit
- **Database**: SQLite
- **Training**: AWS SageMaker (Tesla T4 GPU)

---

## 📦 Datasets Used

| Dataset | Script | Purpose |
|---------|--------|---------|
| CCPD2019 | Chinese | Detection + OCR train/test |
| Roboflow LP v11 | Mixed | Detection train/val/test |
| OpenALPR endtoend | Latin | OCR train |
| UC3M-LP | Latin (Spanish) | OCR train |
| Synthetic (ours) | Latin | OCR train supplement |
| UFPR-ALPR | Latin (Brazilian) | ByteTrack evaluation |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Huydinh1205/smart_parking.git
cd smart_parking
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PARSEQ
```bash
git clone https://github.com/baudm/parseq /tmp/parseq
pip install /tmp/parseq
```

### 5. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
smart_parking/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
└── models/
    ├── yolov11s/
    │      └── yolo11s.pt             # YOLOv11s trained weights
    |__ yolov26s/
    |   └── yolo26s.pt                # YOLOv26s trained weights               
    ├── zero_dce/
    │   └── zero_dce_best.pth       # Zero-DCE++ trained weights
    └── parseq_finetune/
        └── parseq_finetuned_best.pt # Fine-tuned PARSEQ weights
```

---

## 🖥️ App Pages

| Page | Description |
|------|-------------|
| 📷 Upload Image | Upload a photo → detect + read plate → log gate event |
| 🎥 Upload Video | Upload clip → ByteTrack + majority vote → gate events |
| 📸 Live Webcam | Webcam snapshot → detect + read instantly |
| 📊 Results Log | Live dashboard: parked vehicles, fees, revenue |

---

## 📚 References

1. Bautista & Atienza, "PARSEQ: Scene Text Recognition with Permuted Autoregressive Sequence Models," ECCV 2022
2. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022
3. Jocher & Qiu, "Ultralytics YOLO11," 2024
4. Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," CVPR 2020
5. Xu et al., "CCPD: Towards End-to-End License Plate Detection and Recognition," ECCV 2018
6. Roboflow Universe, "License Plate Recognition Dataset v11," 2024