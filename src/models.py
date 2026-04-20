"""Model definitions: transfer learning wrappers for waste classification."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224

# Mapping of supported transfer learning backbones
_BACKBONES = {
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
    "vgg16": (keras.applications.VGG16, keras.applications.vgg16.preprocess_input),
    "mobilenetv2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    "efficientnetb3": (keras.applications.EfficientNetB3, keras.applications.efficientnet.preprocess_input),
}


@keras.utils.register_keras_serializable(package="SmartWaste")
class BackbonePreprocess(layers.Layer):
    """Custom layer that applies backbone-specific preprocessing.

    Unlike Lambda, this serializes/deserializes correctly when saving models.
    """

    def __init__(self, backbone_name, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        if backbone_name not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone_name}'")
        self._preprocess_fn = _BACKBONES[backbone_name][1]

    def call(self, x):
        return self._preprocess_fn(x)

    def get_config(self):
        config = super().get_config()
        config["backbone_name"] = self.backbone_name
        return config


def build_transfer_model(base_name: str, num_classes: int):
    """Build a transfer learning model with frozen backbone.

    The model internally rescales [0,1] → [0,255] then applies the
    backbone-specific preprocessing, so the data pipeline stays universal.

    Parameters
    ----------
    base_name : str – one of 'resnet50', 'vgg16', 'mobilenetv2', 'efficientnetb3'
    num_classes : int

    Returns
    -------
    (model, base_model) tuple – full model and the backbone reference
    """
    if base_name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{base_name}'. Choose from {list(_BACKBONES)}")

    backbone_cls, _ = _BACKBONES[base_name]

    base_model = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Rescale [0,1] → [0,255] then apply backbone preprocessing
    x = layers.Rescaling(255.0)(inputs)
    x = BackbonePreprocess(base_name, name=f"{base_name}_preprocess")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=f"{base_name}_transfer")
    return model, base_model


def unfreeze_layers(model, base_model, num_layers: int = 30):
    """Unfreeze the last *num_layers* of the backbone for fine-tuning.

    All BatchNormalization layers remain frozen to avoid noisy statistics
    with small batch sizes.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    # Keep all BN layers frozen regardless of position
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
