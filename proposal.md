# Smart Waste Vision

**An AI-Based Waste Detection and Classification System Using Convolutional Neural Networks**

---

## 1. Introduction

### 1.1 Background

Waste management remains one of the most pressing environmental challenges in urban Cambodia. Rapid urbanization and population growth in cities like Phnom Penh, Siem Reap, and Battambang have led to a significant increase in solid waste generation. According to various reports, Cambodia generates thousands of tons of municipal waste daily, yet a substantial portion remains unsorted and improperly disposed of. The lack of efficient waste sorting at the source contributes to environmental pollution, health hazards, and missed opportunities for recycling and resource recovery.

Computer vision, a subfield of artificial intelligence, has demonstrated remarkable capabilities in image classification and object detection tasks. With the advancement of Convolutional Neural Networks (CNNs) and transfer learning techniques, it is now feasible to build accurate image classification systems that can identify and categorize different types of waste materials from photographs. This project, Smart Waste Vision, proposes the development of an AI-based waste detection and classification system that leverages deep learning techniques studied in the Computer Vision course to address the waste sorting problem.

### 1.2 Problem Statement

Manual waste sorting is labor-intensive, inconsistent, and often inaccurate. In Cambodia, most waste ends up in landfills without proper segregation, leading to contamination of recyclable materials and environmental degradation. There is a clear need for an automated, scalable, and cost-effective solution that can accurately classify waste into distinct categories to facilitate proper recycling and disposal. The core computer vision challenge lies in building a robust image classifier that can distinguish between visually similar waste categories under varying conditions such as different lighting, angles, backgrounds, and levels of contamination.

### 1.3 Objectives

The primary objectives of this project are:

1. To develop an image classification model using Convolutional Neural Networks (CNNs) that can accurately classify waste images into predefined categories.
2. To evaluate and compare multiple CNN architectures including custom-built models and pre-trained models via transfer learning (e.g., ResNet50, VGG16, MobileNetV2).
3. To apply essential computer vision preprocessing techniques including image resizing, normalization, data augmentation, and feature extraction.
4. To analyze model performance using standard classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
5. To build a simple web-based interface for image upload and real-time waste classification demonstration.

### 1.4 Scope

This project focuses primarily on the image classification task using computer vision techniques. The scope includes dataset preparation and preprocessing, model training and evaluation, and a basic deployment interface. The system will classify waste images into categories defined by the chosen dataset. Hardware-level integration (e.g., robotic sorting arms or embedded camera systems) and real-time video stream processing are outside the scope of this proposal but are noted as potential future extensions.

---

## 2. Dataset

### 2.1 Dataset Source

This project will use the Garbage Classification V2 dataset available on Kaggle (<https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2>). This dataset is a comprehensive image collection specifically curated for garbage classification and recycling research. It contains real-world images of waste items captured in authentic landfill environments, making it highly suitable for training robust classification models.

### 2.2 Dataset Characteristics

The dataset contains images across the following nine waste categories:

| # | Category | Description |
|---|----------|-------------|
| 1 | Cardboard | Cardboard boxes, packaging materials, corrugated sheets |
| 2 | Food Organics | Leftover food, fruit peels, vegetable scraps, organic matter |
| 3 | Glass | Glass bottles, jars, broken glass items |
| 4 | Metal | Aluminum cans, tin cans, metal lids, foil |
| 5 | Miscellaneous Trash | Items that do not fit other categories |
| 6 | Paper | Newspapers, magazines, office paper, receipts |
| 7 | Plastic | Plastic bottles, bags, containers, wrappers |
| 8 | Textile Trash | Clothing scraps, fabric pieces, rags |
| 9 | Vegetation | Leaves, branches, grass clippings, plant waste |

Key dataset specifications include an image resolution of 524 x 524 pixels in color (RGB) format. The images were collected from authentic landfill and waste processing environments, which provides realistic training conditions for the model.

### 2.3 Data Preprocessing Pipeline

The following preprocessing steps will be applied to prepare the dataset for model training:

- **Image Resizing:** All images will be resized to a uniform dimension (224 x 224 pixels) to match the input requirements of standard CNN architectures.
- **Normalization:** Pixel values will be scaled to the [0, 1] range or standardized using ImageNet mean and standard deviation values for transfer learning models.
- **Data Augmentation:** To improve model generalization and address class imbalance, augmentation techniques will be applied including random horizontal and vertical flips, rotation (up to 30 degrees), zoom and crop variations, brightness and contrast adjustments, and Gaussian noise injection.
- **Train-Validation-Test Split:** The dataset will be divided into 70% training, 15% validation, and 15% test sets using stratified sampling to maintain class distribution.
- **Class Balancing:** Oversampling of minority classes or class-weighted loss functions will be employed if significant class imbalance is detected.

---

## 3. Methodology

### 3.1 Overall Approach

The project follows a systematic computer vision pipeline consisting of data collection and preprocessing, feature extraction through convolutional layers, model training with multiple architectures, hyperparameter tuning, evaluation, and deployment. The emphasis is on applying and comparing various CNN-based techniques covered in the Computer Vision course.

### 3.2 Model Architectures

The following architectures will be explored and compared:

**A. Custom CNN (Baseline)**
A CNN model will be built from scratch with multiple convolutional blocks (Conv2D, BatchNormalization, ReLU activation, MaxPooling), followed by fully connected layers with dropout regularization. This serves as the baseline to understand fundamental feature extraction behavior.

**B. ResNet50 (Transfer Learning)**
A ResNet50 model pre-trained on ImageNet will be fine-tuned for waste classification. The residual connections in ResNet help mitigate the vanishing gradient problem and enable training of deeper networks. The final classification layer will be replaced with a 9-class softmax output.

**C. VGG16 (Transfer Learning)**
VGG16 provides a simpler, sequential architecture with small 3x3 convolutional filters. Its straightforward design makes it an excellent reference point for comparing feature extraction approaches against more modern architectures.

**D. MobileNetV2 (Transfer Learning)**
MobileNetV2 uses depthwise separable convolutions and inverted residual blocks, resulting in a lightweight model suitable for mobile and edge deployment scenarios. This is particularly relevant for potential real-world deployment in resource-constrained environments.

### 3.3 Training Configuration

The models will be trained with the following configuration:

- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam with initial learning rate of 0.001
- **Learning Rate Scheduler:** ReduceLROnPlateau with patience of 5 epochs
- **Early Stopping:** Patience of 10 epochs monitoring validation loss
- **Batch Size:** 32
- **Epochs:** Up to 50 (with early stopping)
- **Regularization:** Dropout (0.5) and L2 weight decay

### 3.4 Computer Vision Techniques Applied

This project will demonstrate the application of several key computer vision concepts:

- **Convolutional Feature Extraction:** Learning spatial hierarchies of features from raw pixel data through stacked convolutional layers.
- **Transfer Learning and Fine-Tuning:** Leveraging pre-trained weights from ImageNet and adapting them to the waste classification domain through progressive unfreezing of layers.
- **Data Augmentation:** Applying geometric and photometric transformations to increase training data diversity and improve model robustness.
- **Batch Normalization:** Stabilizing and accelerating training by normalizing intermediate feature maps.
- **Grad-CAM Visualization:** Generating class activation maps to visualize which regions of the input image the model focuses on when making predictions, providing interpretability of the classification decisions.
- **Confusion Matrix Analysis:** Detailed per-class error analysis to identify commonly misclassified categories and understand model limitations.

---

## 4. Evaluation Metrics

Model performance will be rigorously evaluated using the following metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall percentage of correctly classified images across all categories. |
| Precision | Proportion of true positives among all positive predictions for each class (measures false positive rate). |
| Recall | Proportion of true positives among all actual positives for each class (measures false negative rate). |
| F1-Score | Harmonic mean of precision and recall, providing a balanced measure especially useful for imbalanced classes. |
| Confusion Matrix | Visual representation of classification results showing per-class performance and common misclassification patterns. |
| Grad-CAM Maps | Visual heatmaps overlaid on input images to interpret which spatial regions influence the model's predictions. |
| Training Curves | Plots of training and validation accuracy/loss over epochs to analyze convergence, overfitting, and generalization. |

All models will be evaluated on the same held-out test set to ensure fair comparison. A comprehensive comparison table and visualization of results will be included in the final report.

---

## 5. Tools and Technologies

| Tool / Technology | Purpose |
|-------------------|---------|
| Python 3.10+ | Primary programming language for all development |
| TensorFlow / Keras | Deep learning framework for building, training, and evaluating CNN models |
| OpenCV | Image preprocessing, loading, and augmentation operations |
| NumPy / Pandas | Numerical computation and data manipulation |
| Matplotlib / Seaborn | Visualization of training curves, confusion matrices, and Grad-CAM outputs |
| Scikit-learn | Classification metrics computation and dataset splitting utilities |
| Local GPU (RTX 4050) / Jupyter | Development environment with GPU acceleration for model training |
| Streamlit / Gradio | Web interface framework for building the classification demo application |
| Kaggle API | Programmatic dataset download and management |

---

## 6. Project Timeline

The project will follow the timeline outlined below, spanning approximately 8 weeks:

| Week | Tasks | Deliverables |
|------|-------|--------------|
| Week 1 | Project proposal writing, dataset exploration, and literature review | Proposal document |
| Week 2 | Dataset download, exploratory data analysis, and preprocessing pipeline setup | EDA report, cleaned dataset |
| Week 3 | Custom CNN baseline model design and initial training | Baseline model, training logs |
| Week 4 | Transfer learning implementation (ResNet50, VGG16) | Fine-tuned models |
| Week 5 | MobileNetV2 implementation, hyperparameter tuning across all models | Optimized models |
| Week 6 | Comprehensive evaluation, Grad-CAM visualization, and comparative analysis | Evaluation results, visualizations |
| Week 7 | Web interface development and integration with best-performing model | Demo application |
| Week 8 | Final report writing, presentation preparation, and project submission | Final report, presentation, code |

---

## 7. Expected Outcomes

Upon completion, this project is expected to deliver the following outcomes:

1. A trained CNN-based image classifier capable of categorizing waste images into 9 distinct categories with a target accuracy of 85% or higher on the test set.
2. A comparative analysis of at least four CNN architectures (Custom CNN, ResNet50, VGG16, MobileNetV2) with detailed performance metrics and visual explanations.
3. Grad-CAM visualizations demonstrating model interpretability and providing insight into the learned feature representations for each waste category.
4. A functional web-based demo application where users can upload waste images and receive instant classification results with confidence scores.
5. A comprehensive technical report documenting the complete computer vision pipeline, experimental results, and analysis.

---

## 8. Potential Extensions

While outside the current scope, the following extensions are identified for future development:

- **Object Detection:** Extending from single-image classification to multi-object detection using YOLO or Faster R-CNN, enabling the system to detect and count multiple waste items within a single image.
- **Real-Time Video Processing:** Integrating the model with live camera feeds for continuous waste monitoring and classification.
- **Mobile Deployment:** Optimizing the model using TensorFlow Lite for deployment on mobile devices, enabling on-the-go waste classification.
- **Localized Dataset:** Collecting and annotating waste images specific to Cambodian urban environments to build a locally relevant dataset and improve model accuracy for regional waste types.
- **Environmental Impact Dashboard:** Developing a dashboard that aggregates classification data to provide insights on waste composition trends and recycling potential.

---

## 9. Conclusion

Smart Waste Vision presents a practical application of computer vision techniques to address a real-world environmental problem. By leveraging Convolutional Neural Networks and transfer learning, this project aims to demonstrate how image classification can be applied to automate waste sorting and promote recycling efforts. The project provides a comprehensive opportunity to apply core concepts from the Computer Vision course, including image preprocessing, CNN architectures, transfer learning, data augmentation, model evaluation, and visual interpretability through Grad-CAM. The expected outcome is a functional, well-evaluated waste classification system that contributes to environmental sustainability awareness while showcasing the power of modern computer vision techniques.

---

## 10. References

1. Kaggle. Garbage Classification V2 Dataset. <https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2>
2. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. IEEE CVPR, 2016.
3. K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," in Proc. ICLR, 2015.
4. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in Proc. IEEE CVPR, 2018.
5. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in Proc. IEEE ICCV, 2017.
6. J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in Proc. IEEE CVPR, 2009.
