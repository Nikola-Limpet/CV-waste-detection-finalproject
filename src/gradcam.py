"""Grad-CAM visualization: heatmap generation and overlay utilities."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_last_conv_layer(model) -> str:
    """Find the name of the last Conv2D layer, handling nested (transfer) models.

    Returns a '/' delimited path for nested models (e.g. 'resnet50/conv5_block3_3_conv')
    or a simple name for flat models (e.g. 'conv2d_3').
    """
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, keras.layers.Conv2D):
                    last_conv = sub_layer.name
            if last_conv:
                return layer.name + "/" + last_conv
        elif isinstance(layer, keras.layers.Conv2D):
            last_conv = layer.name
    return last_conv


def _build_grad_model(model, last_conv_layer_name):
    """Build a Keras Model that outputs [conv_features, predictions].

    Handles both flat models (custom CNN) and nested models (transfer learning).
    """
    parts = last_conv_layer_name.split("/")

    if len(parts) == 1:
        # Flat model — straightforward
        conv_layer = model.get_layer(last_conv_layer_name)
        return keras.Model(model.input, [conv_layer.output, model.output])

    # Nested model: base model is a sub-model (e.g. ResNet50 inside our wrapper)
    base_name, conv_name = parts
    base = model.get_layer(base_name)
    conv_layer = base.get_layer(conv_name)

    # Strategy: rebuild the outer model replacing the base with an "extended" base
    # that outputs both conv features and its normal output.
    ext_base = keras.Model(base.input, [conv_layer.output, base.output])

    # Iterate through the outer model's layers (simple chain architecture):
    #   Input → Rescaling → Lambda(preprocess) → base_model → GAP → Dense → Dropout → Dense
    inp = keras.Input(shape=model.input_shape[1:])
    x = inp
    conv_out = None
    for layer in model.layers[1:]:  # skip InputLayer
        if layer.name == base_name:
            conv_out, x = ext_base(x, training=False)
        else:
            x = layer(x)

    return keras.Model(inp, [conv_out, x])


def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """Compute Grad-CAM heatmap for the predicted class.

    Parameters
    ----------
    model : keras.Model
    img_array : np.ndarray of shape (1, 224, 224, 3)
    last_conv_layer_name : str – may be 'layer_name' or 'nested/layer_name'

    Returns
    -------
    heatmap : np.ndarray of shape (H, W) in [0, 1]
    predicted_class : int
    """
    grad_model = _build_grad_model(model, last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_class]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(predicted_class)


def overlay_gradcam(img, heatmap, alpha=0.4):
    """Overlay a Grad-CAM heatmap onto an image.

    Parameters
    ----------
    img : np.ndarray – original image (H, W, 3), float [0,1] or uint8
    heatmap : np.ndarray – (h, w) in [0, 1]
    alpha : float – blending factor

    Returns
    -------
    np.ndarray – overlaid image, uint8 RGB
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def generate_gradcam_grid(model, images, labels, class_names,
                          layer_name, save_path=None, cols=4):
    """Generate a grid of Grad-CAM overlays.

    Parameters
    ----------
    images : list of np.ndarray (224, 224, 3) in [0,1]
    labels : list of int (true labels)
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for i in range(n):
        img_array = np.expand_dims(images[i], axis=0)
        heatmap, pred_cls = get_gradcam_heatmap(model, img_array, layer_name)
        overlay = overlay_gradcam(images[i], heatmap)

        axes[i].imshow(overlay)
        true_name = class_names[labels[i]]
        pred_name = class_names[pred_cls]
        color = "green" if labels[i] == pred_cls else "red"
        axes[i].set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=9)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
