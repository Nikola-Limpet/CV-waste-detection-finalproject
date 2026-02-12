# Smart Waste Vision

> CNN-based waste image classification into 9 categories using TensorFlow/Keras

**Dataset:** [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) | **Framework:** TensorFlow 2.x / Keras | **Python:** 3.10+ | **GPU:** NVIDIA RTX 4050

---

## Overview

Smart Waste Vision is an AI-based waste detection and classification system that uses Convolutional Neural Networks to classify waste images into 9 categories. The project compares a custom CNN baseline against three transfer learning models (ResNet50, VGG16, MobileNetV2) and includes Grad-CAM visualizations for model interpretability.

## Dataset

The [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) dataset contains 524x524 RGB images from real landfill environments across 9 classes:

| # | Category | Examples |
|---|----------|----------|
| 1 | Cardboard | Boxes, packaging, corrugated sheets |
| 2 | Food Organics | Fruit peels, vegetable scraps, leftovers |
| 3 | Glass | Bottles, jars, broken glass |
| 4 | Metal | Aluminum cans, tin cans, foil |
| 5 | Miscellaneous Trash | Items not fitting other categories |
| 6 | Paper | Newspapers, magazines, office paper |
| 7 | Plastic | Bottles, bags, containers, wrappers |
| 8 | Textile Trash | Clothing scraps, fabric, rags |
| 9 | Vegetation | Leaves, branches, grass clippings |

## Project Structure

```
smart-waste-vision/
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Preprocessed & split data
│   └── augmented/            # Saved augmented samples
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_gradcam.ipynb
├── src/
│   ├── data_loader.py        # Dataset loading & preprocessing
│   ├── augmentation.py       # Augmentation pipeline
│   ├── models.py             # All model architectures
│   ├── train.py              # Training loop & callbacks
│   ├── evaluate.py           # Metrics & visualization
│   └── gradcam.py            # Grad-CAM implementation
├── app/
│   └── app.py                # Streamlit/Gradio web demo
├── outputs/
│   ├── models/               # Saved model weights
│   ├── plots/                # Training curves, confusion matrices
│   └── gradcam/              # Grad-CAM visualizations
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (developed on RTX 4050, 6 GB VRAM)
- CUDA Toolkit 11.8+ and cuDNN 8.6+

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/smart-waste-vision.git
cd smart-waste-vision

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API key)
kaggle datasets download -d sumn2u/garbage-classification-v2 -p data/raw/
unzip data/raw/garbage-classification-v2.zip -d data/raw/
```

### Verify GPU

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should show your RTX 4050
```

## Usage

```bash
# 1. Exploratory Data Analysis
jupyter notebook notebooks/01_eda.ipynb

# 2. Train baseline CNN
jupyter notebook notebooks/02_baseline_cnn.ipynb

# 3. Train transfer learning models (ResNet50, VGG16, MobileNetV2)
jupyter notebook notebooks/03_transfer_learning.ipynb

# 4. Evaluate and compare all models
jupyter notebook notebooks/04_evaluation.ipynb

# 5. Grad-CAM visualizations
jupyter notebook notebooks/05_gradcam.ipynb

# 6. Launch web demo
python app/app.py
```

## Models

| Model | Type | Parameters | Description |
|-------|------|------------|-------------|
| Custom CNN | Baseline | ~1M | 4-block CNN built from scratch |
| ResNet50 | Transfer Learning | ~23M | Fine-tuned on waste classification |
| VGG16 | Transfer Learning | ~138M | Sequential architecture reference |
| MobileNetV2 | Transfer Learning | ~3.4M | Lightweight, deployment-friendly |

## Results

| Model | Accuracy | Macro F1 | Weighted F1 | Inference (ms) |
|-------|----------|----------|-------------|----------------|
| Custom CNN | — | — | — | — |
| ResNet50 | — | — | — | — |
| VGG16 | — | — | — | — |
| MobileNetV2 | — | — | — | — |

*Results will be populated after training.*

## References

1. [Garbage Classification V2 Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
2. He et al., "Deep Residual Learning for Image Recognition," IEEE CVPR, 2016
3. Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR, 2015
4. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," IEEE CVPR, 2018
5. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," IEEE ICCV, 2017
6. [Intelligent Waste Sorting — Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-08461-w)
7. [AI-Powered Waste Classification Using CNNs (2024)](https://thesai.org/Downloads/Volume15No10/Paper_9-AI_Powered_Waste_Classification.pdf)
# CV-waste-detection-finalproject
