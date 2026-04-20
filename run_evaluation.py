"""Evaluate all 4 models and generate comparison plots.

Equivalent to running notebook 04_evaluation.ipynb.
"""

import os
import sys
import json

import matplotlib
matplotlib.use("Agg")

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/yuujin/school/cv-final-project/.cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from src.data_loader import load_and_split_data, create_dataset
from src.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_model_comparison,
    get_misclassified_samples, measure_inference_time,
)
import src.models  # registers BackbonePreprocess

DATA_DIR = os.path.join("data", "raw", "standardized_256")
MODELS_DIR = os.path.join("outputs", "models")
PLOTS_DIR = os.path.join("outputs", "plots")

# Load data
train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = \
    load_and_split_data(DATA_DIR)
test_ds = create_dataset(test_paths, test_labels, augment=False, shuffle=False)
print(f"Test set: {len(test_paths)} images, {len(class_names)} classes")

# Load and evaluate all models
MODEL_NAMES = ["resnet50", "vgg16", "mobilenetv2", "efficientnetb3"]
models = {}
all_results = {}

for name in MODEL_NAMES:
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if os.path.exists(path):
        print(f"\nLoading {name}...")
        model = tf.keras.models.load_model(path)
        models[name] = model
        print(f"  Params: {model.count_params():,}")

        print(f"  Evaluating on test set...")
        results = evaluate_model(model, test_ds, class_names)
        results["params"] = model.count_params()
        results["inference_ms"] = measure_inference_time(model, test_ds)
        all_results[name] = results

        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Inference: {results['inference_ms']:.1f} ms/image")
        print(results["report"])
    else:
        print(f"WARNING: {path} not found — skipping")

# All confusion matrices in 2x2 grid
print("\nGenerating confusion matrix grid...")
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()
for idx, (name, results) in enumerate(all_results.items()):
    if idx >= 4:
        break
    sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
    axes[idx].set_title(f"{name} (acc={results['accuracy']:.3f})", fontsize=12)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("True")
for j in range(len(all_results), 4):
    axes[j].axis("off")
plt.suptitle("Confusion Matrices — All Models", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "all_confusion_matrices.png"), dpi=150)
plt.close()

# Comparison table
print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
comparison_data = []
for name, res in all_results.items():
    comparison_data.append({
        "Model": name,
        "Test Accuracy": f"{res['accuracy']:.4f}",
        "Parameters": f"{res['params']:,}",
        "Inference (ms)": f"{res['inference_ms']:.1f}",
    })
df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))

# Comparison bar chart
print("\nGenerating comparison chart...")
plot_model_comparison(all_results, save_path=os.path.join(PLOTS_DIR, "model_comparison.png"))

# Misclassification analysis (best model)
best_name = max(all_results, key=lambda k: all_results[k]["accuracy"])
best_model = models[best_name]
print(f"\nBest model: {best_name} ({all_results[best_name]['accuracy']:.4f})")

misclassified = get_misclassified_samples(best_model, test_ds, class_names, n=8)
if misclassified:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, (img, true_cls, pred_cls) in enumerate(misclassified):
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls}", color="red", fontsize=9)
        axes[i].axis("off")
    for j in range(len(misclassified), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Misclassified Samples — {best_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "misclassified.png"), dpi=150)
    plt.close()

# Save results as JSON for later use
results_json = {}
for name, res in all_results.items():
    results_json[name] = {
        "accuracy": res["accuracy"],
        "params": res["params"],
        "inference_ms": res["inference_ms"],
    }
with open(os.path.join("outputs", "evaluation_results.json"), "w") as f:
    json.dump(results_json, f, indent=2)

print("\nDone! All evaluation results saved.")
