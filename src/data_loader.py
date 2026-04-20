"""Dataset loading, splitting, and tf.data pipeline construction."""

import os
import pathlib

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.augmentation import get_augmentation_layer

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def discover_class_names(data_dir: str) -> list[str]:
    """Return sorted list of subdirectory names (one per class)."""
    data_path = pathlib.Path(data_dir)
    return sorted([d.name for d in data_path.iterdir() if d.is_dir()])


def load_and_split_data(data_dir: str, seed: int = 42):
    """Walk *data_dir*, collect image paths + integer labels, stratified split.

    Returns
    -------
    (train_paths, train_labels,
     val_paths, val_labels,
     test_paths, test_labels,
     class_names)
    """
    class_names = discover_class_names(data_dir)
    label_map = {name: idx for idx, name in enumerate(class_names)}

    paths, labels = [], []
    for cls_name in class_names:
        cls_dir = os.path.join(data_dir, cls_name)
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if os.path.isfile(fpath):
                paths.append(fpath)
                labels.append(label_map[cls_name])

    paths = np.array(paths)
    labels = np.array(labels)

    # 70 / 15 / 15 stratified split (two-step)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=seed,
    )

    return (train_paths, train_labels,
            val_paths, val_labels,
            test_paths, test_labels,
            class_names)


def _load_image(path, label):
    """Read, decode, resize, and normalize a single image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0  # normalize to [0, 1]
    return img, label


def create_dataset(paths, labels, augment=False, batch_size=BATCH_SIZE, shuffle=True,
                    cache=True):
    """Build a tf.data.Dataset pipeline.

    Parameters
    ----------
    paths : array-like of file path strings
    labels : array-like of integer labels
    augment : bool – apply training augmentation
    batch_size : int
    shuffle : bool
    cache : bool – cache decoded images in memory (disable for large datasets)
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))

    ds = ds.batch(batch_size)

    if augment:
        aug_layer = get_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y),
                     num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_class_weights(labels) -> dict:
    """Compute balanced class weights for handling class imbalance."""
    unique = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique, y=labels)
    return dict(zip(unique, weights))


def preprocess_single_image(image: np.ndarray) -> np.ndarray:
    """Preprocess a single RGB image for model inference.

    Parameters
    ----------
    image : np.ndarray – HxWx3 RGB image (uint8 or float)

    Returns
    -------
    np.ndarray – (1, 224, 224, 3) normalized to [0, 1]
    """
    img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0).numpy()
