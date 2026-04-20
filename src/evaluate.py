"""Evaluation utilities: metrics, confusion matrices, training curves, comparison."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PLOTS_DIR = os.path.join("outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate_model(model, test_ds, class_names: list) -> dict:
    """Run model on test set and return metrics dict."""
    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred,
                                        target_names=class_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix",
                          save_path=None):
    """Seaborn heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_training_curves(history, title="Training Curves", save_path=None):
    """1x2 subplot: accuracy + loss over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["accuracy"], label="Train")
    ax1.plot(history["val_accuracy"], label="Val")
    ax1.set_title(f"{title} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["loss"], label="Train")
    ax2.plot(history["val_loss"], label="Val")
    ax2.set_title(f"{title} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


def merge_histories(h1, h2) -> dict:
    """Concatenate two Keras history dicts (for 2-phase training plots)."""
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged


def plot_model_comparison(results_dict: dict, save_path=None):
    """Grouped bar chart comparing accuracy across models."""
    models = list(results_dict.keys())
    accuracies = [results_dict[m]["accuracy"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=sns.color_palette("viridis", len(models)))
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


def get_misclassified_samples(model, test_ds, class_names, n=5):
    """Return the first *n* misclassified (image, true, pred) tuples."""
    misclassified = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        for i in range(len(labels)):
            true_label = labels[i].numpy()
            pred_label = pred_labels[i]
            if true_label != pred_label and len(misclassified) < n:
                misclassified.append((
                    images[i].numpy(),
                    class_names[true_label],
                    class_names[pred_label],
                ))
        if len(misclassified) >= n:
            break
    return misclassified


def measure_inference_time(model, test_ds, num_runs=100) -> float:
    """Average inference time in milliseconds per single image."""
    # Grab one batch and take the first image
    for images, _ in test_ds.take(1):
        single = images[:1]
        break

    # Warm-up
    model.predict(single, verbose=0)

    start = time.time()
    for _ in range(num_runs):
        model.predict(single, verbose=0)
    elapsed = time.time() - start

    return (elapsed / num_runs) * 1000  # ms
