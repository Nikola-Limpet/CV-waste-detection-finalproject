"""Generate Grad-CAM visualizations for the best model.

Equivalent to running notebook 05_gradcam.ipynb.
"""

import os

import matplotlib
matplotlib.use("Agg")

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/yuujin/school/cv-final-project/.cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from src.data_loader import load_and_split_data, create_dataset
from src.gradcam import (
    find_last_conv_layer, get_gradcam_heatmap, overlay_gradcam, generate_gradcam_grid,
)
from src.evaluate import get_misclassified_samples
import src.models  # registers BackbonePreprocess

DATA_DIR = os.path.join("data", "raw", "standardized_256")
MODELS_DIR = os.path.join("outputs", "models")
GRADCAM_DIR = os.path.join("outputs", "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Load best model
MODEL_PRIORITY = ["efficientnetb3", "resnet50", "mobilenetv2", "vgg16"]
model = None
model_name = None
for name in MODEL_PRIORITY:
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        model_name = name
        print(f"Loaded: {name}")
        break

if model is None:
    raise FileNotFoundError("No trained models found")

# Load data
train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = \
    load_and_split_data(DATA_DIR)
test_ds = create_dataset(test_paths, test_labels, augment=False, shuffle=False, cache=False)

# Find last conv layer
layer_name = find_last_conv_layer(model)
print(f"Last conv layer: {layer_name}")

# --- Correctly classified samples (2 per class) ---
print("\nCollecting correctly classified samples...")
correct_images, correct_labels = [], []
counts = {i: 0 for i in range(len(class_names))}
target_per_class = 2

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    for i in range(len(labels)):
        true_label = int(labels[i].numpy())
        if pred_labels[i] == true_label and counts[true_label] < target_per_class:
            correct_images.append(images[i].numpy())
            correct_labels.append(true_label)
            counts[true_label] += 1
    if all(c >= target_per_class for c in counts.values()):
        break

print(f"Collected {len(correct_images)} correctly classified samples")

print("Generating Grad-CAM grid (correct predictions)...")
generate_gradcam_grid(
    model, correct_images, correct_labels, class_names, layer_name,
    save_path=os.path.join(GRADCAM_DIR, "correct_predictions.png"),
    cols=4,
)

# --- Misclassified samples ---
print("Collecting misclassified samples...")
misclassified = get_misclassified_samples(model, test_ds, class_names, n=8)

if misclassified:
    mis_images = [m[0] for m in misclassified]
    mis_true_labels = [class_names.index(m[1]) for m in misclassified]

    print("Generating Grad-CAM grid (incorrect predictions)...")
    generate_gradcam_grid(
        model, mis_images, mis_true_labels, class_names, layer_name,
        save_path=os.path.join(GRADCAM_DIR, "incorrect_predictions.png"),
        cols=4,
    )
else:
    print("No misclassified samples found!")

# --- Cross-class comparison ---
print("Generating cross-class comparison...")
if len(correct_images) > 0:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(min(4, len(correct_images))):
        img = correct_images[i]
        img_array = np.expand_dims(img, axis=0)
        heatmap, pred_cls = get_gradcam_heatmap(model, img_array, layer_name)
        overlay = overlay_gradcam(img, heatmap)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Original: {class_names[correct_labels[i]]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Grad-CAM: {class_names[pred_cls]}", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Original vs Grad-CAM Overlay", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(GRADCAM_DIR, "cross_class_comparison.png"), dpi=150)
    plt.close()

print(f"\nDone! Grad-CAM visualizations saved to {GRADCAM_DIR}/")
print(f"  correct_predictions.png")
print(f"  incorrect_predictions.png")
print(f"  cross_class_comparison.png")
