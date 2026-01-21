# ***INTRODUCTION*** for using my project: Resistors Leads Recognition

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-eb1616?logo=ultralytics&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Roboflow](https://img.shields.io/badge/Annotation-Roboflow-6706ce?logo=roboflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository hosts a state-of-the-art Deep Learning-based Quality Control (QC) system specifically engineered for the automated inspection of resistor leads. Leveraging a fine-tuned YOLOv8 Nano Segmentation architecture, the system provides real-time detection and defect analysis, ensuring high precision in industrial inspection environments. 

The project is designed for seamless deployment via a **Streamlit** web interface, allowing for both high-resolution image analysis and live webcam testing.

Annotation Tool: **Roboflow** at https://roboflow.com/

Model Framework: **Ultralytics YOLOv8n-seg**

I still update the model to enhance its efficiency!

## I. System Overview

**The system operates through an end-to-end Computer Vision pipeline:**
* Inference Engine: Utilizes the **`best_nano_seed_1301.pt`** weights to perform object detection on resistor leads.
* Logic-Based Assessment: Analyzes detection results to classify the product status:
    * **`Lead_good`**: Leads that meet quality standards.
    * **`Lead_bad`**: Any lead is identified as "bad" (e.g., bent, broken, or missing).
* Defect Visualization: Automatically crops and zooms into detected anomalies for manual verification by QC personnel.

## II. Project Structure

Below is a breakdown of the repository's components as seen in the project root:

| File/Folder | Description |
| :--- | :--- |
| **`Graph_best_nano_seed_1301/`** | Contains performance metrics, including Loss curves, mAP (Mean Average Precision) graphs, and Confusion Matrices for model validation. |
| **`app.py`** | The main application entry point. A Streamlit-based UI for model interaction and visualization. |
| **`best_nano_seed_1301.pt`** | The optimized YOLOv8 Nano model weights, fine-tuned for high sensitivity to small electronic components. |
| **`resistor_legs.yaml`** | The dataset configuration file defining paths and class names (`Lead_good`, `Lead_bad`) used during training. |
| **`requirements.txt`** | A comprehensive list of Python dependencies (e.g., `ultralytics`, `streamlit`, `opencv-python`). |
| **`.gitignore`** / **`LICENSE`** | Standard repository configuration and MIT License documentation. |
| **`README.md`** | This documentation file. |

## III. Prerequisites

Ensure your environment meets the following technical specifications:
* **Python: 3.10 or 3.11**
* **Hardware:** An **NVIDIA GPU** is recommended for faster inference, though the "Nano" model performs efficiently on CPUs.
* **Camera:** A high-definition webcam for the live testing module.

## IV. Installation Guide

### 1. Clone the Repository:
`git clone https://github.com/duyanhtoanvui/Resistor-Legs-Recognition.git
cd Resistor-Legs-Recognition`

### 2. Set Up Virtual Environment (Recommended)
Isolate your dependencies to avoid conflicts.

**Windows:**
`python -m venv venv
.\venv\Scripts\activate`

**macOS / Linux:**
`python3 -m venv venv
source venv/bin/activate`

### 3. Install Dependencies:
`pip install -r requirements.txt`

## V. Usage & App Operations

To launch the QC interface, execute:
`streamlit run app.py`

**Operating the Dashboard**:
* *Configuration (Sidebar):*
  * Confidence Threshold: Adjust the sensitivity of the model. Recommended: 0.15 for webcams, 0.25 for high-quality uploads first, then increase later.
  * IoU Threshold: Controls how the model handles overlapping bounding boxes.
* *Input Methods:*
  * Tab 1 (Upload): Best for high-resolution industrial photos. Supports JPG and PNG.
  * Tab 2 (Webcam): Designed for quick, live testing. Ensure the component is 5-10cm from the lens with adequate lighting.
* *Result Analysis:*
  * The Conclusion Banner will instantly display "PRODUCT PASSED" or "DEFECTS DETECTED".
  * Auto-Zoom: If a defect is found, the system generates a "Defect Details" grid, showing specifically where the failure occurred.
  * Statistics: Expand the "Detailed Statistics" section to view a quantitative breakdown of detected leads.

### VI. Troubleshooting
* `ModuleNotFoundError`: Ensure your virtual environment is active and you have run `pip install -r requirements.txt`.
* Model Not Loading: Verify that `best_nano_seed_1301.pt` is located in the root directory alongside `app.py`.
* Low Detection Accuracy: If the model misses leads, try lowering the **Confidence Threshold** in the sidebar or improving the ambient lighting of your workspace.

### VII. Author: Nguyen Tan Duy Anh - Ho Chi Minh University of Science (HCMUS)
