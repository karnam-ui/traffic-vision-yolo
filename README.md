# 🚗 TrafficVision — Real-Time Highway Object Detection & Tracking

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

A real-time computer vision pipeline engineered for multi-class vehicular detection, highway traffic flow tracking, and dynamic lighting contrast enhancement using **YOLO11** and **CLAHE** (Contrast Limited Adaptive Histogram Equalization).

---

## 🎯 Project Overview

Accurate traffic monitoring on high-speed transit corridors is frequently impaired by harsh lighting variances, headlights, shadows, and weather atmospheric distortion. 

**TrafficVision** integrates:
- **Contrast Enhancement**: CLAHE transforms grayscale luminance tiles ($8 \times 8$ grid, clip limit $2.0$) to prevent shadow loss without amplifying high-frequency noise.
- **YOLO11 Nano Inference**: Utilizes Ultralytics YOLO11 (`yolo11n.pt`) tuned for ultra-low latency inference, high frame rates ($>60$ FPS on modern GPUs), and reliable multi-scale vehicle localization (cars, trucks, buses, motorcycles).
- **High-Resolution Evaluation**: Evaluated across 4K transit corridor video footage with multi-object IoU non-maximum suppression ($0.75$) and confidence thresholding ($0.65$).

---

## 🔬 Computer Vision Pipeline

```mermaid
flowchart LR
    A["Raw Video Frame / 4K Stream"] --> B["Grayscale / BGR Split"]
    B --> C["CLAHE Enhancement<br/>(clipLimit=2.0, tileGrid=(8,8))"]
    C --> D["YOLO11n Neural Detector<br/>(imgsz=1280, conf=0.65, iou=0.75)"]
    D --> E["Bounding Box Annotation & Tracking"]
    E --> F["Rendered Visual Output (cv2.imshow)"]
```

---

## 📁 Repository Structure

```
├── yolo11n.pt           # Pre-trained YOLO11 Nano weights
├── vidtrial.py          # Real-time video detection and bounding box visualization
├── clahetrial.py        # Video inference coupled with CLAHE contrast preprocessing
├── imagetrial.py        # Static image detection test harness
├── classes.py           # Object class definitions and taxonomy
└── README.md
```

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.9+
- Webcam or video file

### 1. Installation
```bash
git clone https://github.com/karnam-ui/traffic-vision-yolo.git
cd traffic-vision-yolo
pip install ultralytics opencv-python
```

### 2. Run Video Inference
```bash
# Standard detection
python vidtrial.py

# Contrast-enhanced CLAHE detection
python clahetrial.py
```

Press `q` at any time to exit the video display stream.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
