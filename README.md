# Smart Waste Vision

> CNN-based waste image classification into 10 categories using TensorFlow/Keras

**Dataset:** [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) | **Framework:** TensorFlow 2.x / Keras | **Python:** 3.10+ | **GPU:** NVIDIA RTX 4050

---

## Overview

Smart Waste Vision is an AI-based waste detection and classification system that uses Convolutional Neural Networks to classify waste images into 10 categories. The project compares four transfer learning models (EfficientNet-B3, ResNet50, VGG16, MobileNetV2) and includes Grad-CAM visualizations for model interpretability.

## Dataset

The [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) dataset contains 13,347 RGB images (standardized to 256x256) across 10 waste categories:

| # | Category | Examples |
|---|----------|----------|
| 1 | Battery | AA batteries, button cells, rechargeable packs |
| 2 | Biological | Food scraps, fruit peels, organic waste |
| 3 | Cardboard | Boxes, packaging, corrugated sheets |
| 4 | Clothes | Clothing items, fabric scraps |
| 5 | Glass | Bottles, jars, broken glass |
| 6 | Metal | Aluminum cans, tin cans, foil |
| 7 | Paper | Newspapers, magazines, office paper |
| 8 | Plastic | Bottles, bags, containers, wrappers |
| 9 | Shoes | Sneakers, boots, sandals |
| 10 | Trash | General waste not fitting other categories |

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

# 2. Train transfer learning models (ResNet50, VGG16, MobileNetV2, EfficientNet-B3)
jupyter notebook notebooks/03_transfer_learning.ipynb

# 3. Evaluate and compare all models
jupyter notebook notebooks/04_evaluation.ipynb

# 4. Grad-CAM visualizations
jupyter notebook notebooks/05_gradcam.ipynb

# 5. Launch web demo
python app/app.py
```

## Models

| Model | Type | Parameters | Description |
|-------|------|------------|-------------|
| EfficientNet-B3 | Transfer Learning | 11.2M | Compound-scaled architecture, best accuracy |
| ResNet50 | Transfer Learning | 24.1M | Deep residual connections |
| VGG16 | Transfer Learning | 14.8M | Sequential architecture reference |
| MobileNetV2 | Transfer Learning | 2.6M | Lightweight, deployment-friendly |

## Results

| Model | Test Accuracy | Parameters | Inference (ms) |
|-------|-------------|------------|----------------|
| **EfficientNet-B3** | **93.96%** | 11.2M | 55.0 |
| ResNet50 | 82.93% | 24.1M | 53.4 |
| MobileNetV2 | 81.73% | 2.6M | 52.2 |
| VGG16 | 80.33% | 14.8M | 54.5 |

EfficientNet-B3 significantly outperforms all other models, achieving 93.96% test accuracy with 2-phase transfer learning (freeze backbone → fine-tune last 40 layers). All models were trained with ImageNet-pretrained weights on the 10-class waste dataset (70/15/15 train/val/test split).

## References

1. [Garbage Classification V2 Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
2. Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019
3. He et al., "Deep Residual Learning for Image Recognition," IEEE CVPR, 2016
4. Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR, 2015
5. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," IEEE CVPR, 2018
6. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," IEEE ICCV, 2017
7. [Intelligent Waste Sorting — Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-08461-w)
8. [AI-Powered Waste Classification Using CNNs (2024)](https://thesai.org/Downloads/Volume15No10/Paper_9-AI_Powered_Waste_Classification.pdf)

