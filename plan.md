# Smart Waste Vision — Implementation Plan

## Project Overview

**Goal:** Build a CNN-based image classification system that classifies waste images into 9 categories using the Garbage Classification V2 dataset from Kaggle.

**Dataset:** [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- 9 classes: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation
- Image resolution: 524×524 (RGB)

**Tech Stack:** Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit/Gradio

---

## Phase 1: Environment Setup & Dataset Preparation

### 1.1 Environment Setup
- Set up local environment with Python virtual environment and CUDA support
  - **GPU:** NVIDIA RTX 4050 (6 GB VRAM) — may need to reduce batch size for VGG16 or use `tf.config.experimental.set_memory_growth`
  - **CUDA Toolkit:** 11.8+ with cuDNN 8.6+
  - **Python:** 3.10+ in a `venv` virtual environment
- Install required packages: `tensorflow`, `opencv-python`, `matplotlib`, `seaborn`, `scikit-learn`, `gradio`
- Create project folder structure:
  ```
  smart-waste-vision/
  ├── data/
  │   ├── raw/                  # Original dataset
  │   ├── processed/            # Preprocessed & split data
  │   └── augmented/            # (Optional) saved augmented samples
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
  │   ├── models/               # Saved model weights (.h5 / .keras)
  │   ├── plots/                # Training curves, confusion matrices
  │   └── gradcam/              # Grad-CAM visualizations
  ├── requirements.txt
  └── README.md
  ```

### 1.2 Dataset Download & Organization
- Download dataset via Kaggle API: `kaggle datasets download -d sumn2u/garbage-classification-v2`
- Unzip and verify folder structure (each class in its own subfolder)
- Verify image counts per class
- Check for corrupted or unreadable images (use OpenCV to try loading each)

### 1.3 Exploratory Data Analysis (EDA)
- **Class distribution:** Bar chart showing number of images per category — identify any class imbalance
- **Sample visualization:** Display a grid of 3–5 random images per class to understand visual characteristics
- **Image properties:** Analyze image dimensions, aspect ratios, color channel distributions
- **Per-class analysis:** Mean pixel intensity per class, color histogram comparison
- **Similarity check:** Identify visually similar categories that may be hard to distinguish (e.g., Paper vs. Cardboard, Textile vs. Miscellaneous)

**Deliverable:** `01_eda.ipynb` with all visualizations and written observations

---

## Phase 2: Data Preprocessing Pipeline

### 2.1 Image Preprocessing (`data_loader.py`)
- Resize all images to **224×224** pixels (standard input for pretrained CNNs)
- Normalize pixel values to [0, 1] range
- For transfer learning models: apply ImageNet standardization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 2.2 Dataset Splitting
- Split into **70% train / 15% validation / 15% test**
- Use `sklearn.model_selection.train_test_split` with `stratify` parameter to maintain class proportions
- Save the split file paths to CSV or use `tf.keras.utils.image_dataset_from_directory` with `validation_split`

### 2.3 Data Augmentation (`augmentation.py`)
- Apply augmentation **only to training set** using `tf.keras.layers` or `ImageDataGenerator`:
  - Random horizontal flip
  - Random rotation (±20°)
  - Random zoom (0.8–1.2)
  - Random brightness adjustment (±0.2)
  - Random contrast adjustment
  - Optional: Gaussian noise
- Visualize augmented samples side-by-side with originals to verify transformations look reasonable

### 2.4 Class Imbalance Handling
- Calculate class weights using `sklearn.utils.class_weight.compute_class_weight`
- Pass `class_weight` dict to `model.fit()`
- Alternative: Oversample minority classes using augmentation

### 2.5 Data Pipeline
- Use `tf.data.Dataset` pipeline for efficient loading:
  - `.cache()` → `.shuffle()` → `.batch(32)` → `.prefetch(tf.data.AUTOTUNE)`
- This ensures fast GPU utilization during training

**Deliverable:** `data_loader.py`, `augmentation.py`, verified data pipeline

---

## Phase 3: Model Development

### 3.1 Model A — Custom CNN Baseline (`models.py`)

Architecture design:
```
Input (224×224×3)
  → Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  → Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  → Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  → Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  → GlobalAveragePooling2D
  → Dense(256) → Dropout(0.5) → ReLU
  → Dense(9, softmax)
```

Purpose: Establish a baseline accuracy to compare against transfer learning models.

### 3.2 Model B — ResNet50 (Transfer Learning)
- Load `tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))`
- Freeze base model layers initially
- Add custom head: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(9, softmax)
- **Phase 1:** Train only the head (5–10 epochs)
- **Phase 2:** Unfreeze last 20–30 layers of ResNet50, reduce learning rate to 1e-5, fine-tune (10–20 epochs)

### 3.3 Model C — VGG16 (Transfer Learning)
- Load `tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))`
- Same two-phase training strategy as ResNet50
- Note: VGG16 is heavier (138M params) — expect slower training

### 3.4 Model D — MobileNetV2 (Transfer Learning)
- Load `tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))`
- Same two-phase training strategy
- Note: Lightweight model (~3.4M params) — fastest inference, good for deployment comparison

### 3.5 Training Configuration (`train.py`)
- **Optimizer:** Adam (lr=1e-3 for head training, lr=1e-5 for fine-tuning)
- **Loss:** `CategoricalCrossentropy` (or `SparseCategoricalCrossentropy` depending on label format)
- **Callbacks:**
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)`
  - `ModelCheckpoint(save_best_only=True)`
  - `TensorBoard` for live monitoring (optional)
- **Batch size:** 32 (reduce to 16 if GPU memory is constrained on the RTX 4050, especially for VGG16)
- **Max epochs:** 50 (early stopping will likely trigger earlier)

**Deliverable:** `02_baseline_cnn.ipynb`, `03_transfer_learning.ipynb`, saved model weights in `outputs/models/`

---

## Phase 4: Evaluation & Analysis

### 4.1 Quantitative Metrics (`evaluate.py`)
For each of the 4 models, compute on the **test set**:
- Overall accuracy
- Per-class precision, recall, F1-score (use `sklearn.metrics.classification_report`)
- Macro and weighted average F1-score
- Confusion matrix

### 4.2 Comparison Table
Create a summary table:

| Model | Params | Accuracy | Macro F1 | Weighted F1 | Inference Time (ms) |
|-------|--------|----------|----------|-------------|---------------------|
| Custom CNN | ? | ? | ? | ? | ? |
| ResNet50 | ? | ? | ? | ? | ? |
| VGG16 | ? | ? | ? | ? | ? |
| MobileNetV2 | ? | ? | ? | ? | ? |

### 4.3 Visualizations
- **Confusion matrices:** Heatmap for each model using `seaborn.heatmap`
- **Training curves:** Accuracy and loss plots (train vs. validation) for each model
- **Per-class accuracy comparison:** Grouped bar chart across all 4 models
- **Misclassification analysis:** Display top misclassified image pairs (predicted vs. actual) to understand failure cases

### 4.4 Grad-CAM Visualization (`gradcam.py`)
- Implement Grad-CAM for the best-performing model
- Target the last convolutional layer
- Steps:
  1. Forward pass to get predictions
  2. Compute gradients of the predicted class w.r.t. feature maps
  3. Global average pool the gradients
  4. Weight feature maps by pooled gradients
  5. Apply ReLU and overlay heatmap on original image
- Generate Grad-CAM visualizations for 2–3 correctly classified and 2–3 misclassified images per category
- This demonstrates **model interpretability** — a key CV concept

**Deliverable:** `04_evaluation.ipynb`, `05_gradcam.ipynb`, all plots saved to `outputs/plots/` and `outputs/gradcam/`

---

## Phase 5: Web Demo Application

### 5.1 Application Design (`app/app.py`)
- Use **Gradio** (simpler) or **Streamlit** (more customizable)
- Features:
  - Image upload (drag-and-drop or file picker)
  - Display uploaded image preview
  - Run inference using the best model
  - Show predicted category with confidence score
  - Display a bar chart of top-3 prediction probabilities
  - Show Grad-CAM heatmap overlay for the prediction
  - Display category description (what it is, how to recycle it)

### 5.2 Model Optimization for Inference
- Save the best model in `.keras` or SavedModel format
- Optional: Convert to TFLite for faster inference
- Ensure preprocessing (resize, normalize) matches training pipeline exactly

### 5.3 Testing the App
- Test with images from the test set
- Test with new images from the internet (out-of-distribution)
- Test with edge cases: blurry images, multiple objects, unusual angles

**Deliverable:** Working `app.py`, screenshot/demo recording

---

## Phase 6: Documentation & Final Report

### 6.1 Code Documentation
- Add docstrings to all functions in `src/`
- Comment key sections in notebooks
- Write `README.md` with setup instructions, usage guide, and results summary

### 6.2 Final Report Structure
1. Introduction & Problem Statement
2. Related Work (cite the reference papers)
3. Dataset Description & EDA
4. Methodology (preprocessing, augmentation, architectures)
5. Experimental Setup (hyperparameters, hardware, training details)
6. Results & Analysis (metrics table, confusion matrices, training curves)
7. Grad-CAM Interpretability Analysis
8. Web Demo Description
9. Conclusion & Future Work
10. References

### 6.3 Presentation
- Prepare 10–15 slides covering key points
- Include live demo of the web application
- Highlight the CV techniques used and what was learned

---

## Implementation Checklist

### Phase 1: Setup & Data
- [ ] Set up local environment (venv, CUDA, cuDNN) with RTX 4050 GPU
- [ ] Download and organize dataset
- [ ] Verify all images load correctly
- [ ] Complete EDA notebook with class distribution, sample images, observations

### Phase 2: Preprocessing
- [ ] Implement image resizing and normalization
- [ ] Split dataset (70/15/15) with stratification
- [ ] Implement augmentation pipeline
- [ ] Calculate class weights
- [ ] Build efficient `tf.data` pipeline

### Phase 3: Models
- [ ] Build and train Custom CNN baseline
- [ ] Implement and fine-tune ResNet50
- [ ] Implement and fine-tune VGG16
- [ ] Implement and fine-tune MobileNetV2
- [ ] Save all model weights

### Phase 4: Evaluation
- [ ] Compute metrics for all 4 models
- [ ] Generate confusion matrices
- [ ] Plot training curves
- [ ] Create model comparison table
- [ ] Implement Grad-CAM
- [ ] Generate Grad-CAM visualizations
- [ ] Analyze misclassifications

### Phase 5: Demo
- [ ] Build Gradio/Streamlit web interface
- [ ] Integrate best model for inference
- [ ] Add Grad-CAM overlay in the app
- [ ] Test with diverse images

### Phase 6: Documentation
- [ ] Write README.md
- [ ] Complete final report
- [ ] Prepare presentation slides

---

## Key References

1. Kaggle Dataset: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
2. ResNet50 Garbage Classification (Kaggle Notebook): https://www.kaggle.com/code/sumn2u/garbage-classification-resnet
3. Transfer Learning Notebook (same dataset): https://www.kaggle.com/code/sumn2u/garbage-classification-transfer-learning
4. DenseNet Notebook (same dataset): https://www.kaggle.com/code/sumn2u/garbage-classification-densenet-tl
5. ResNet50 Scratch to Transfer Learning (GitHub): https://github.com/FarzadNekouee/Garbage_Classification_ResNet50_Scratch_to_Transfer-Learning
6. Intelligent waste sorting — Scientific Reports (2025): https://www.nature.com/articles/s41598-025-08461-w
7. Enhancing trash classification using federated DL — Scientific Reports (2024): https://www.nature.com/articles/s41598-024-62003-4
8. AI-Powered Waste Classification Using CNNs (2024): https://thesai.org/Downloads/Volume15No10/Paper_9-AI_Powered_Waste_Classification.pdf