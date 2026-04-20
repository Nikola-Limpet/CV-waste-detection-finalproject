# Smart Waste Vision — Presentation Evaluation Report

**Course:** Computer Vision Final Project  
**Project:** Smart Waste Vision — CNN-Based Waste Image Classification  
**Date:** April 2026  
**Target Accuracy:** 85% | **Achieved:** 93.96% (EfficientNet-B3)

---

## Evaluation Criteria Summary

| # | Criteria | Weight | Self-Score | Weighted |
|---|----------|--------|------------|----------|
| 1 | Introduction | 10% | 9/10 | 9.0% |
| 2 | Literature Review | 15% | 13/15 | 13.0% |
| 3 | Dataset | 15% | 14/15 | 14.0% |
| 4 | Methodology | 15% | 14/15 | 14.0% |
| 5 | Experiments and Results | 15% | 14/15 | 14.0% |
| 6 | Demo | 10% | 10/10 | 10.0% |
| 7 | Overall Presentation | 10% | 9/10 | 9.0% |
| 8 | Q&A | 10% | 9/10 | 9.0% |
| | **Total** | **100%** | | **92.0%** |

---

## 1. Introduction (10%) — *2 minutes*

### 1.1 Problem Context

Waste management is one of the most pressing environmental challenges in urban Cambodia. Rapid urbanization in cities like Phnom Penh, Siem Reap, and Battambang has led to a significant increase in solid waste generation — thousands of tons daily. A substantial portion remains unsorted and improperly disposed of, contributing to:

- **Environmental pollution** — contamination of soil, water, and air
- **Health hazards** — disease vectors from untreated organic waste
- **Missed recycling opportunities** — recyclable materials lost to landfills

Manual waste sorting is labor-intensive, inconsistent, and often inaccurate. There is a clear need for an automated, scalable, and cost-effective solution.

### 1.2 Computer Vision Solution

Convolutional Neural Networks (CNNs) and transfer learning techniques now make it feasible to build accurate image classifiers that can identify and categorize different types of waste materials from photographs. **Smart Waste Vision** applies these CV course concepts to address the real-world waste sorting problem.

### 1.3 Project Objectives

1. Develop a CNN image classifier for **10 waste categories**
2. Compare **4 transfer learning architectures**: EfficientNet-B3, ResNet50, VGG16, MobileNetV2
3. Apply CV preprocessing techniques: resizing, normalization, augmentation, feature extraction
4. Evaluate using standard metrics + **Grad-CAM** for model interpretability
5. Build a functioning **web demo** for real-time waste classification

### 1.4 Scope

- **In scope:** Image classification, model comparison, Grad-CAM visualization, web demo
- **Out of scope:** Object detection/segmentation, hardware integration, real-time video stream
- **Target:** 85%+ test accuracy

### Score Justification: 9/10

Clear real-world problem grounded in Cambodian context. Five measurable objectives. Appropriate scope for a CV course project with honest boundaries.

---

## 2. Literature Review (15%) — *2.5 minutes*

### 2.1 Foundational CNN Architectures

| Paper | Year | Key Contribution | Relevance to Project |
|-------|------|-----------------|---------------------|
| He et al. — **ResNet** | 2016 | Residual/skip connections | Solves vanishing gradient problem; enables training 50+ layer networks |
| Simonyan & Zisserman — **VGG16** | 2015 | Sequential 3x3 convolutions | Simpler architecture serving as a performance baseline |
| Sandler et al. — **MobileNetV2** | 2018 | Depthwise separable convolutions + inverted residual blocks | Lightweight model for resource-constrained deployment (relevant to Cambodia) |
| Tan & Le — **EfficientNet** | 2019 | Compound scaling (depth + width + resolution) | Added beyond original proposal; achieves superior accuracy with fewer parameters |

### 2.2 Model Interpretability

| Paper | Year | Key Contribution | Relevance to Project |
|-------|------|-----------------|---------------------|
| Selvaraju et al. — **Grad-CAM** | 2017 | Gradient-weighted class activation maps | Visual explanations of model predictions; builds trust in automated waste sorting |

### 2.3 Transfer Learning Foundation

| Paper | Year | Key Contribution | Relevance to Project |
|-------|------|-----------------|---------------------|
| Deng et al. — **ImageNet** | 2009 | Large-scale hierarchical image database | Foundation for all pretrained weights used in transfer learning |

### 2.4 Domain-Specific Work

| Paper | Year | Key Finding |
|-------|------|------------|
| Intelligent Waste Sorting — *Scientific Reports* | 2025 | Validates active research in AI-based waste classification |
| AI-Powered Waste Classification Using CNNs — *IJACSA* | 2024 | Confirms CNN approach effectiveness for waste categorization |

### 2.5 Why Transfer Learning Works for Waste Classification

ImageNet does not contain waste classes, but **low-level features** learned from ImageNet (edges, textures, color patterns, shapes) transfer effectively across visual domains. The pretrained backbone acts as a powerful general-purpose feature extractor, and fine-tuning the higher layers specializes the model for waste recognition.

**Key decision:** EfficientNet-B3 was added after initial experiments showed ResNet50, VGG16, and MobileNetV2 plateauing at 80-83%. Compound scaling (simultaneously optimizing depth, width, and resolution) promised a better accuracy-efficiency tradeoff — and delivered 93.96%.

### Score Justification: 13/15

8+ references spanning foundational architectures, interpretability, and recent domain-specific work. Each reference connected to a specific project design decision. Minor gap: could include additional waste-specific ML papers.

---

## 3. Dataset (15%) — *2.5 minutes*

### 3.1 Dataset Source

**Garbage Classification V2** from Kaggle (`sumn2u/garbage-classification-v2`)
- Real-world images captured in authentic landfill environments
- 13,347 RGB images standardized to 256x256 pixels
- 10 waste categories

### 3.2 Class Distribution

| # | Category | Disposal Type | Color Code |
|---|----------|--------------|------------|
| 1 | Battery | Hazardous | Red |
| 2 | Biological | Compostable | Green |
| 3 | Cardboard | Recyclable | Blue |
| 4 | Clothes | Recyclable | Blue |
| 5 | Glass | Recyclable | Blue |
| 6 | Metal | Recyclable | Blue |
| 7 | Paper | Recyclable | Blue |
| 8 | Plastic | Recyclable | Blue |
| 9 | Shoes | Landfill | Gray |
| 10 | Trash | Landfill | Gray |

### 3.3 Data Split Strategy

| Set | Samples | Percentage |
|-----|---------|------------|
| Training | 9,343 | 70% |
| Validation | 2,002 | 15% |
| Test | 2,003 | 15% |

- **Stratified split** using `sklearn.model_selection.train_test_split` with `stratify` parameter
- Two-step split: first 70/30, then 30 into 50/50 for val/test
- **Seed = 42** for full reproducibility

### 3.4 Preprocessing Pipeline

| Step | Implementation | Details |
|------|---------------|---------|
| Resize | `tf.image.resize` | 224x224 (standard CNN input) |
| Normalize | `Rescaling(1./255)` | Pixel values to [0, 1] |
| Backbone Preprocessing | Custom `BackbonePreprocess` layer | Model-specific ImageNet standardization |
| Data Pipeline | `tf.data.Dataset` | `.cache().shuffle().batch(32).prefetch(AUTOTUNE)` |

**Engineering highlight:** The `BackbonePreprocess` layer is a custom Keras layer registered with `@keras.utils.register_keras_serializable()`. This avoids Lambda serialization issues and makes models fully self-contained for deployment.

### 3.5 Data Augmentation (Training Only)

| Augmentation | Range | Implementation |
|-------------|-------|---------------|
| Horizontal Flip | 50% probability | `RandomFlip("horizontal")` |
| Rotation | ±20 degrees | `RandomRotation(0.055)` |
| Zoom | ±20% | `RandomZoom(0.2)` |
| Brightness | ±20% | `RandomBrightness(0.2)` |
| Contrast | ±20% | `RandomContrast(0.2)` |

Augmentation is applied **only to training data** via the `tf.data` pipeline mapping — validation and test sets remain unaugmented.

### 3.6 Class Imbalance Handling

```python
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
```

Balanced class weights are passed directly to `model.fit(class_weight=...)`, adjusting the loss function to penalize misclassifications of minority classes more heavily — without requiring oversampling that could cause overfitting.

### Score Justification: 14/15

Proper stratification, efficient tf.data pipeline, appropriate augmentation, class weight handling, and custom serializable preprocessing layer. Minor gap: dataset contains Western waste images, not Cambodian-specific.

---

## 4. Methodology (15%) — *3 minutes*

### 4.1 Model Architectures

All four models share the same **unified transfer learning architecture**:

```
Input (224x224x3)
  → Rescaling (×255.0)
  → BackbonePreprocess (model-specific ImageNet standardization)
  → Frozen Backbone (pretrained on ImageNet)
  → GlobalAveragePooling2D
  → Dense(256, activation='relu')
  → Dropout(0.5)
  → Dense(10, activation='softmax')
```

| Model | Parameters | Key Innovation | Layers Unfrozen (Phase 2) |
|-------|-----------|----------------|--------------------------|
| EfficientNet-B3 | 11.2M | Compound scaling (depth + width + resolution) | 40 |
| ResNet50 | 24.1M | Residual/skip connections | 30 |
| VGG16 | 14.8M | Sequential 3x3 convolutions | 8 |
| MobileNetV2 | 2.6M | Depthwise separable convolutions + inverted residuals | 30 |

### 4.2 Two-Phase Transfer Learning Strategy

**Phase 1 — Head Training (10 epochs)**
- Freeze entire backbone (pretrained weights locked)
- Train only the classification head (GAP + Dense layers)
- Learning rate: 1e-3 (Adam optimizer)
- Purpose: Learn to map backbone features to 10 waste categories

**Phase 2 — Fine-Tuning (20 epochs)**
- Unfreeze last N layers of the backbone (varies per model)
- Reduce learning rate to 1e-5 (prevents catastrophic forgetting)
- **Critical:** All BatchNormalization layers remain frozen
  - Reason: With batch sizes of 32 (or 16 for VGG16), batch statistics would be noisy and unrepresentative. Keeping BN frozen preserves stable ImageNet statistics.

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | SparseCategoricalCrossentropy |
| Optimizer | Adam |
| Phase 1 Learning Rate | 1e-3 |
| Phase 2 Learning Rate | 1e-5 |
| Batch Size | 32 (16 for VGG16) |
| EarlyStopping | patience=10, restore_best_weights=True |
| ReduceLROnPlateau | factor=0.5, patience=5 |
| ModelCheckpoint | save_best_only=True |
| GPU | NVIDIA RTX 4050 (6 GB VRAM) |
| Framework | TensorFlow 2.20.0 |

**Practical engineering note:** VGG16 required batch size 16 due to 6GB VRAM constraint. Models were trained in separate subprocesses (`run_remaining_training.py`) to fully release GPU memory between training runs.

### 4.4 Grad-CAM Implementation

Grad-CAM provides visual explanations of model predictions:

1. **Forward pass** through the model to the last convolutional layer
2. **Compute gradients** of the predicted class score w.r.t. feature maps using `tf.GradientTape`
3. **Global average pool** gradients to get importance weights per channel
4. **Weighted sum** of feature maps by their importance weights
5. **ReLU activation** (keep only positive influences)
6. **Normalize** to [0, 1], resize to input dimensions
7. **Overlay** as a color heatmap using `cv2.applyColorMap(COLORMAP_JET)` with alpha blending

**Implementation handles both flat CNN and nested transfer learning models** via `find_last_conv_layer()` function that traverses model architectures recursively.

### Score Justification: 14/15

Four diverse architectures with different design philosophies. Two-phase transfer learning with frozen BatchNorm. Custom serializable preprocessing. Per-model hyperparameter tuning. Automatic Grad-CAM for nested models. Minor gap: manual hyperparameter tuning rather than formal search (grid/random/Bayesian).

---

## 5. Experiments and Results (15%) — *3.5 minutes*

### 5.1 Main Results

| Model | Test Accuracy | Parameters | Inference (ms/image) |
|-------|:------------:|:----------:|:-------------------:|
| **EfficientNet-B3** | **93.96%** | **11.2M** | **83.2** |
| ResNet50 | 82.93% | 24.1M | 58.0 |
| MobileNetV2 | 81.73% | 2.6M | 79.4 |
| VGG16 | 80.33% | 14.8M | 57.7 |

**Target: 85% accuracy → Achieved: 93.96% (exceeded by ~9 percentage points)**

### 5.2 Efficiency Analysis

| Model | Accuracy per Million Params | Verdict |
|-------|:-------------------------:|---------|
| MobileNetV2 | 31.4% / M | Most parameter-efficient |
| EfficientNet-B3 | 8.4% / M | Best absolute accuracy |
| VGG16 | 5.4% / M | Moderate efficiency |
| ResNet50 | 3.4% / M | Least parameter-efficient |

### 5.3 Key Findings

1. **EfficientNet-B3 dominates** — 11% accuracy gap over the second-best model (ResNet50). Compound scaling (simultaneously scaling depth, width, and resolution) is the differentiator.

2. **More parameters ≠ better accuracy** — VGG16 (14.8M params) performs worst despite having 5x more parameters than MobileNetV2 (2.6M). Architecture design matters more than raw parameter count.

3. **MobileNetV2 is deployment-optimal** — highest accuracy-per-parameter ratio at 31.4%/M, making it the best choice for mobile/edge deployment in resource-constrained environments.

4. **All models achieve real-time inference** — under 100ms per image, making interactive classification feasible.

5. **Common confusion patterns** — visually similar categories cause most errors: Clothes ↔ Shoes, Paper ↔ Cardboard, general Trash ↔ specific categories.

### 5.4 Visualizations Generated

| Visualization | File | Content |
|--------------|------|---------|
| Model Comparison | `outputs/plots/model_comparison.png` | Bar chart comparing all 4 models |
| Confusion Matrices | `outputs/plots/all_confusion_matrices.png` | 2x2 grid showing per-class performance |
| Training Curves | `outputs/plots/efficientnetb3_curves.png` | Accuracy & loss over epochs (shows 2-phase training) |
| Misclassified Samples | `outputs/plots/misclassified.png` | 8 failure cases from best model |

### 5.5 Grad-CAM Analysis

| Visualization | File | Insight |
|--------------|------|---------|
| Correct Predictions | `outputs/gradcam/correct_predictions.png` | Model focuses on relevant object features (texture, shape, material) |
| Incorrect Predictions | `outputs/gradcam/incorrect_predictions.png` | Model attends to background textures or ambiguous regions |
| Cross-Class Comparison | `outputs/gradcam/cross_class_comparison.png` | Side-by-side original vs. heatmap overlay for top 4 classes |

**Grad-CAM insight:** On correctly classified images, the model consistently attends to discriminative object features (bottle shape for glass, woven texture for clothes). On misclassified images, attention often drifts to background elements, suggesting that background context contributes to errors on visually similar classes.

### Score Justification: 14/15

Exceeds 85% accuracy target by ~9 points. Multi-dimensional evaluation: accuracy, parameter count, inference time, confusion matrices, training curves, misclassification analysis, efficiency ratios, and Grad-CAM interpretability. Minor gap: per-class precision/recall/F1 table not prominently featured in outputs.

---

## 6. Demo (10%) — *3 minutes*

### 6.1 Platform

**Gradio web application** — 900+ lines of code with custom CSS styling  
**Deployment:** Docker container on Railway (port 7860)

### 6.2 Features — 6 Tabs

| Tab | Feature | Description |
|-----|---------|-------------|
| 1. **Classify** | Image Classification | Upload or webcam capture → top-5 confidence scores + Grad-CAM heatmap + disposal guidance + AI advice |
| 2. **Live Webcam** | Real-Time Detection | Auto-detect every 2 seconds with annotated output and top-3 confidence bars |
| 3. **Batch Processing** | Multi-Image Analysis | Upload multiple images → DataFrame results → CSV export with summary statistics |
| 4. **AI Chat** | EcoBot Assistant | Conversational waste disposal Q&A powered by Gemini 2.5 Flash Lite (via OpenRouter) |
| 5. **History & Stats** | Session Dashboard | Pie charts by category/disposal type, diversion metrics, environmental impact summary |
| 6. **Waste Guide** | Educational Reference | Searchable guide for all 10 categories: disposal steps, decomposition times, recycling facts, safety alerts |

### 6.3 Additional Features

- **Model switching dropdown** — compare all 4 models live during the demo
- **Confidence-aware UX** — warnings for low confidence (<40%) and medium confidence (<70%) predictions
- **Structured waste knowledge base** — disposal instructions, environmental impact data, safety alerts for all categories
- **Color-coded bin system** — Blue (recyclable), Green (compostable), Red (hazardous), Gray (landfill)
- **AI-powered advice** — context-aware recycling tips from Gemini after each classification

### 6.4 Demo Flow for Presentation (3 minutes)

| Time | Action |
|------|--------|
| 0:00-0:30 | Show the UI, briefly walk through all 6 tabs |
| 0:30-1:30 | Upload a waste image → show classification result + Grad-CAM + disposal guidance + AI advice |
| 1:30-2:00 | Webcam: classify a real object in the room |
| 2:00-2:30 | Batch process 3-4 images, show CSV export |
| 2:30-2:45 | Quick chat with EcoBot about a disposal question |
| 2:45-3:00 | Flash History tab showing accumulated session statistics |

### Score Justification: 10/10

Far exceeds the "basic web interface" requirement. Six feature-rich tabs with AI integration, real-time webcam support, batch processing with export, comprehensive educational knowledge base, confidence-aware UX, and production-quality styling. This is a production-grade application, not a course demo.

---

## 7. Overall Presentation (10%) — *18 minutes total*

### 7.1 Presentation Timing

| Segment | Duration | Criteria |
|---------|----------|----------|
| Introduction | 2 min | Introduction (10%) |
| Literature Review | 2.5 min | Literature Review (15%) |
| Dataset | 2.5 min | Dataset (15%) |
| Methodology | 3 min | Methodology (15%) |
| Experiments & Results | 3.5 min | Experiments & Results (15%) |
| Live Demo | 3 min | Demo (10%) |
| Conclusion & Future Work | 1.5 min | Overall Presentation (10%) |
| **Total** | **18 min** | + 2 min Q&A buffer |

### 7.2 Narrative Arc

**Problem** (Cambodia waste crisis) → **Solution** (CV/CNN approach) → **How** (dataset + methodology) → **Proof** (results + Grad-CAM) → **Impact** (demo + real-world applicability)

### 7.3 Slide Design Recommendations

- Clean, consistent design with green/blue gradient theme (`#2E7D32` to `#1565C0`)
- Every slide must include at least one visual element (diagram, chart, screenshot, or Grad-CAM image)
- Minimize text — bullet points only, no paragraphs
- Use the model comparison table and Grad-CAM images as visual anchors

### 7.4 Conclusion Slide Content

**Achievements:**
- 93.96% test accuracy (EfficientNet-B3) — exceeded 85% target
- 4 transfer learning architectures compared with multi-dimensional analysis
- Grad-CAM interpretability showing what the model "sees"
- 6-tab production-quality web demo with AI integration

**Limitations (be transparent):**
- Single-object classification only (not multi-object detection)
- Dataset contains Western waste images, not Cambodian-specific
- Clean/isolated objects differ from real-world dirty/mixed waste conditions

**Future Work:**
- YOLO/Faster R-CNN for multi-object detection
- TensorFlow Lite for mobile deployment (MobileNetV2 baseline ready)
- Local Cambodian waste dataset collection
- Real-time video stream processing
- Environmental impact tracking dashboard

### Score Justification: 9/10

Professional narrative flow, clear structure, appropriate pacing for 18 minutes. Honest about limitations while highlighting significant achievements. Visual-heavy slides recommended.

---

## 8. Q&A Preparation (10%)

### Anticipated Questions & Prepared Answers

---

**Q1: "Why did you add EfficientNet-B3 when it wasn't in the original proposal?"**

After initial experiments, ResNet50, VGG16, and MobileNetV2 all plateaued at 80-83% accuracy. EfficientNet-B3's compound scaling approach (simultaneously scaling depth, width, and resolution) was designed to achieve better accuracy with fewer parameters. The results validated this decision: 93.96% with only 11.2M parameters, compared to ResNet50's 82.93% with 24.1M parameters.

---

**Q2: "Why is there no custom CNN baseline?"**

We prioritized adding a fourth transfer learning architecture (EfficientNet-B3) over a custom CNN baseline. Comparing four architectures with fundamentally different design philosophies — residual connections (ResNet), sequential convolutions (VGG), depthwise separable (MobileNet), and compound scaling (EfficientNet) — provides more insightful comparisons than including a custom baseline that would obviously underperform all pretrained models.

---

**Q3: "How do you handle class imbalance in the dataset?"**

We compute balanced class weights using `sklearn.utils.class_weight.compute_class_weight("balanced")` and pass them to `model.fit(class_weight=...)`. This adjusts the loss function to penalize misclassifications of minority classes more heavily, without requiring oversampling that could lead to overfitting on duplicated minority samples.

---

**Q4: "Why do you keep BatchNormalization layers frozen during fine-tuning?"**

With batch sizes of 32 (or 16 for VGG16), the per-batch statistics computed during fine-tuning would be noisy and unrepresentative of the true data distribution. Keeping BatchNorm layers frozen preserves the stable running mean and variance learned during pretraining on ImageNet's millions of images. This is critical for training stability — unfreezing BN with small batches often causes training to diverge.

---

**Q5: "What are the main failure modes of the model?"**

The primary failure mode is confusion between visually similar categories: Clothes and Shoes (both fabric/leather), Paper and Cardboard (similar texture and color), and general Trash versus specific categories. Grad-CAM analysis on misclassified samples reveals the model sometimes focuses on background textures or ambiguous edges rather than discriminative object features.

---

**Q6: "Could this system work in real-world Cambodia?"**

The foundation is solid, but real-world deployment requires several modifications:
- **(a)** A Cambodian-specific dataset with local waste items and conditions
- **(b)** Object detection (YOLO/Faster R-CNN) for images with multiple waste items
- **(c)** TFLite optimization for mobile deployment (MobileNetV2 is already lightweight at 2.6M params)
- **(d)** Handling dirty, contaminated, or partially occluded items
- **(e)** Integration with local waste management systems and recycling infrastructure

---

**Q7: "Why did you choose Gradio over Streamlit for the web demo?"**

Gradio offers built-in support for image upload, webcam capture, and ML model serving with minimal boilerplate code. The Blocks API enables complex multi-tab layouts (we built 6 tabs). Gradio is also natively deployable on Hugging Face Spaces and supports Docker containers. For an ML-focused demo with image/webcam inputs, Gradio requires significantly less code than Streamlit.

---

**Q8: "Can you explain how Grad-CAM works technically?"**

1. Perform a forward pass to get the last convolutional layer's feature maps
2. Compute gradients of the predicted class score with respect to those feature maps using `tf.GradientTape`
3. Global average pool the gradients across spatial dimensions to get one importance weight per channel
4. Compute a weighted combination of the feature maps using these importance weights
5. Apply ReLU (keep only features with positive influence on the prediction)
6. Normalize to [0, 1], resize to the original input dimensions
7. Apply a color map (JET) and overlay on the original image with alpha blending

The key insight is that the gradients tell us how much each feature map channel matters for the predicted class, and the feature maps tell us where in the image those features are activated.

---

**Q9: "What would you do differently if you started over?"**

- Include a formal hyperparameter search (grid or Bayesian optimization) instead of manual tuning
- Collect a small Cambodian waste dataset for domain adaptation testing
- Add per-class precision/recall/F1 analysis more prominently
- Experiment with ensemble methods (combining predictions from multiple models)
- Try data augmentation strategies like CutMix or MixUp

---

**Q10: "Why are inference times different across models if they all use the same pipeline?"**

Inference time depends on model complexity (number of operations, not just parameters). EfficientNet-B3 uses compound scaling which increases computational depth, hence its 83.2ms. VGG16 and ResNet50 are around 58ms despite different parameter counts because VGG16's sequential architecture is highly optimized on GPU. MobileNetV2's depthwise separable convolutions, while parameter-efficient, involve more memory-bound operations, resulting in 79.4ms despite having the fewest parameters.

### Score Justification: 9/10

10 prepared answers covering architecture choices, technical details, failure modes, real-world applicability, and self-reflection. All answers grounded in actual implementation details from the codebase.

---

## Appendix: Key Project Files

| File | Purpose |
|------|---------|
| `src/models.py` | 4 architecture definitions, BackbonePreprocess custom layer, layer unfreezing logic |
| `src/train.py` | Two-phase training loop, callbacks (EarlyStopping, LR reduction, checkpointing) |
| `src/data_loader.py` | Stratified data split, tf.data pipeline, image preprocessing |
| `src/augmentation.py` | Training-only augmentation pipeline (flip, rotate, zoom, brightness, contrast) |
| `src/evaluate.py` | Metrics computation, confusion matrices, training curves, model comparison |
| `src/gradcam.py` | Grad-CAM implementation for flat and nested model architectures |
| `app/app.py` | 6-tab Gradio web demo (900+ lines) |
| `app/waste_knowledge.py` | Structured waste knowledge base for all 10 categories |
| `app/ai_advisor.py` | Gemini AI integration for contextual waste disposal advice |
| `outputs/evaluation_results.json` | Official benchmark results (accuracy, params, inference time) |
| `outputs/plots/` | Training curves, confusion matrices, model comparison charts |
| `outputs/gradcam/` | Grad-CAM visualizations (correct, incorrect, cross-class comparison) |
| `proposal.md` | Original project proposal |
| `plan.md` | Implementation plan |
| `Dockerfile` | Docker deployment configuration |

---

## References

1. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," IEEE CVPR, 2016.
2. K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR, 2015.
3. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," IEEE CVPR, 2018.
4. M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019.
5. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," IEEE ICCV, 2017.
6. J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," IEEE CVPR, 2009.
7. "Intelligent Waste Sorting," *Scientific Reports*, 2025.
8. "AI-Powered Waste Classification Using CNNs," *IJACSA*, Vol. 15, No. 10, 2024.
9. Garbage Classification V2 Dataset, Kaggle. https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
