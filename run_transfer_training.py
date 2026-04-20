"""Standalone script to train all 3 transfer learning models.

Equivalent to running notebook 03_transfer_learning.ipynb.
Logs progress to stdout (redirect to file with nohup).
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless training

# CUDA setup — must be before TF import
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/yuujin/school/cv-final-project/.cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from src.data_loader import load_and_split_data, create_dataset, get_class_weights
from src.models import build_transfer_model
from src.train import get_callbacks, train_model, fine_tune_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves, merge_histories

DATA_DIR = os.path.join("data", "raw", "standardized_256")
PLOTS_DIR = os.path.join("outputs", "plots")

CONFIGS = {
    "resnet50":    {"unfreeze": 30, "batch_size": 32},
    "vgg16":       {"unfreeze": 8,  "batch_size": 16},
    "mobilenetv2": {"unfreeze": 30, "batch_size": 32},
}

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20


def main():
    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    # Load data once
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = \
        load_and_split_data(DATA_DIR)
    num_classes = len(class_names)
    class_weights = get_class_weights(train_labels)

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # Pre-build default batch size datasets
    train_ds_32 = create_dataset(train_paths, train_labels, augment=True, batch_size=32)
    val_ds_32 = create_dataset(val_paths, val_labels, augment=False, shuffle=False, batch_size=32)

    all_results = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  TRAINING: {name.upper()}")
        print(f"{'='*60}")
        t0 = time.time()

        tf.keras.backend.clear_session()

        # Use correct batch size datasets
        if cfg["batch_size"] != 32:
            t_ds = create_dataset(train_paths, train_labels, augment=True, batch_size=cfg["batch_size"])
            v_ds = create_dataset(val_paths, val_labels, augment=False, shuffle=False, batch_size=cfg["batch_size"])
        else:
            t_ds, v_ds = train_ds_32, val_ds_32

        model, base_model = build_transfer_model(name, num_classes)

        # Phase 1: Train classification head
        print(f"\n--- Phase 1: Training classification head (lr=1e-3) ---")
        callbacks = get_callbacks(name)
        h1 = train_model(model, t_ds, v_ds, PHASE1_EPOCHS, class_weights, callbacks, lr=1e-3)

        # Phase 2: Fine-tune backbone
        print(f"\n--- Phase 2: Fine-tuning last {cfg['unfreeze']} layers (lr=1e-5) ---")
        callbacks = get_callbacks(name)
        h2 = fine_tune_model(model, base_model, t_ds, v_ds, PHASE2_EPOCHS,
                             class_weights, callbacks, num_layers=cfg["unfreeze"], lr=1e-5)

        # Training curves
        merged = merge_histories(h1, h2)
        plot_training_curves(merged, title=name.upper(),
                             save_path=os.path.join(PLOTS_DIR, f"{name}_curves.png"))

        # Evaluate on test set
        te_ds = create_dataset(test_paths, test_labels, augment=False, shuffle=False, batch_size=cfg["batch_size"])
        results = evaluate_model(model, te_ds, class_names)
        all_results[name] = results

        elapsed = time.time() - t0
        print(f"\n{name} Test Accuracy: {results['accuracy']:.4f}")
        print(f"{name} Training time: {elapsed/60:.1f} min")
        print(results["report"])

        plot_confusion_matrix(results["confusion_matrix"], class_names,
                              title=f"{name} — Confusion Matrix",
                              save_path=os.path.join(PLOTS_DIR, f"{name}_cm.png"))

    # Summary
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {name:15s} — Accuracy: {res['accuracy']:.4f}")

    print("\nDone! All models saved to outputs/models/")


if __name__ == "__main__":
    main()
