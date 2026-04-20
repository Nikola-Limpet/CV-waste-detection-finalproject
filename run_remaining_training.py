"""Train VGG16 and MobileNetV2 (ResNet50 already done).

Runs each model in a subprocess to ensure GPU memory is fully released.
"""

import os
import subprocess
import sys

MODELS_TO_TRAIN = [
    ("vgg16", 8, 16),       # (name, unfreeze_layers, batch_size)
    ("mobilenetv2", 30, 32),
]


def train_single_model(name, unfreeze, batch_size):
    """Spawn a subprocess to train one model."""
    script = f"""
import os, sys, time
import matplotlib
matplotlib.use("Agg")
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

name = "{name}"
unfreeze = {unfreeze}
batch_size = {batch_size}

t0 = time.time()

train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = \\
    load_and_split_data(DATA_DIR)
num_classes = len(class_names)
class_weights = get_class_weights(train_labels)

print(f"Classes ({{num_classes}}): {{class_names}}")
print(f"Train: {{len(train_paths)}} | Val: {{len(val_paths)}} | Test: {{len(test_paths)}}")

t_ds = create_dataset(train_paths, train_labels, augment=True, batch_size=batch_size)
v_ds = create_dataset(val_paths, val_labels, augment=False, shuffle=False, batch_size=batch_size)

model, base_model = build_transfer_model(name, num_classes)

print(f"\\n--- Phase 1: Training classification head (lr=1e-3) ---")
callbacks = get_callbacks(name)
h1 = train_model(model, t_ds, v_ds, 10, class_weights, callbacks, lr=1e-3)

print(f"\\n--- Phase 2: Fine-tuning last {{unfreeze}} layers (lr=1e-5) ---")
callbacks = get_callbacks(name)
h2 = fine_tune_model(model, base_model, t_ds, v_ds, 20, class_weights, callbacks,
                     num_layers=unfreeze, lr=1e-5)

merged = merge_histories(h1, h2)
plot_training_curves(merged, title=name.upper(),
                     save_path=os.path.join(PLOTS_DIR, f"{{name}}_curves.png"))

te_ds = create_dataset(test_paths, test_labels, augment=False, shuffle=False, batch_size=batch_size)
results = evaluate_model(model, te_ds, class_names)

elapsed = time.time() - t0
print(f"\\n{{name}} Test Accuracy: {{results['accuracy']:.4f}}")
print(f"{{name}} Training time: {{elapsed/60:.1f}} min")
print(results["report"])

plot_confusion_matrix(results["confusion_matrix"], class_names,
                      title=f"{{name}} — Confusion Matrix",
                      save_path=os.path.join(PLOTS_DIR, f"{{name}}_cm.png"))
print(f"\\nDONE: {{name}}")
"""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {name.upper()} (subprocess)")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd="/home/yuujin/school/cv-final-project",
    )
    if result.returncode != 0:
        print(f"ERROR: {name} training failed with code {result.returncode}")
        return False
    return True


if __name__ == "__main__":
    for name, unfreeze, batch_size in MODELS_TO_TRAIN:
        success = train_single_model(name, unfreeze, batch_size)
        if not success:
            print(f"Stopping due to failure in {name}")
            break

    print("\n" + "="*60)
    print("  ALL TRAINING COMPLETE")
    print("="*60)
    models_dir = os.path.join("outputs", "models")
    for f in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size_mb:.1f} MB")
