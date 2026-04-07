"""
Grad-CAM — Class Activation Map visualisation
=============================================
Keras-3 compatible implementation that works with nested Functional sub-models
(EfficientNetB0 wrapped inside an outer Functional model).

Strategy: run the backbone standalone to get feature maps, then run a
          tiny head-only model to get the prediction — both in one GradientTape.

Public API:
    get_gradcam(model, img_array) -> heatmap (H, W) in [0, 1]
    heatmap_to_bbox(heatmap, img_size, threshold) -> (x1,y1,x2,y2) or None
    overlay_gradcam(image, heatmap, alpha) -> uint8 RGB image
"""

import numpy as np
import tensorflow as tf
import cv2

_BACKBONE_LAYER = "efficientnetb0"
# Augmentation / preprocessing layers that run before the backbone
_PRE_LAYERS = {"aug_flip", "aug_rot", "aug_zoom", "aug_contrast",
               "aug_translate", "efficientnet_preprocess"}


def _build_pre_model(model: tf.keras.Model) -> tf.keras.Model:
    """Return a model that applies just the pre-processing layers."""
    inp = model.input
    x   = inp
    for layer in model.layers:
        if layer.name in _PRE_LAYERS:
            x = layer(x)
    return tf.keras.Model(inputs=inp, outputs=x, name="pre_model")


def _build_head_model(model: tf.keras.Model) -> tf.keras.Model:
    """Return a model: backbone_output -> final prediction."""
    backbone = model.get_layer(_BACKBONE_LAYER)
    inp      = tf.keras.Input(shape=backbone.output_shape[1:], name="feat_input")
    x        = inp
    past_backbone = False
    for layer in model.layers:
        if layer.name == _BACKBONE_LAYER:
            past_backbone = True
            continue
        if past_backbone and layer.name not in _PRE_LAYERS:
            x = layer(x)
    return tf.keras.Model(inputs=inp, outputs=x, name="head_model")


def get_gradcam(model: tf.keras.Model,
                img_array: np.ndarray) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a binary sigmoid classifier.

    Parameters
    ----------
    model     : trained Keras model (EfficientNetB0_BrainTumor)
    img_array : preprocessed image batch (1, H, W, 3) float32

    Returns
    -------
    heatmap : np.ndarray of shape (7, 7) with values in [0, 1]
    """
    backbone = model.get_layer(_BACKBONE_LAYER)

    # Build pre-processing chain once
    pre_model  = _build_pre_model(model)
    head_model = _build_head_model(model)

    x_pre = tf.cast(pre_model(img_array, training=False), tf.float32)

    with tf.GradientTape() as tape:
        x_feat = backbone(x_pre, training=False)
        tape.watch(x_feat)
        preds = head_model(x_feat, training=False)
        loss  = preds[:, 0]

    grads    = tape.gradient(loss, x_feat)         # (1, h, w, c)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)
    conv_out = x_feat[0]                           # (h, w, c)
    heatmap  = conv_out @ pooled[..., tf.newaxis]  # (h, w, 1)
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0)
    max_val  = tf.math.reduce_max(heatmap)
    heatmap  = heatmap / (max_val + 1e-8)

    return heatmap.numpy()


def heatmap_to_bbox(heatmap: np.ndarray,
                    img_size: tuple[int, int],
                    threshold: float = 0.40) -> tuple[int, int, int, int] | None:
    """
    Convert a Grad-CAM heatmap to a tight axis-aligned bounding box.

    Parameters
    ----------
    heatmap   : (h_feat, w_feat) array in [0, 1]
    img_size  : (height, width) of the original image
    threshold : activation threshold (0-1)

    Returns
    -------
    (x1, y1, x2, y2) in original image coordinates, or None
    """
    h, w = img_size
    hmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    binary = (hmap >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_pts = np.concatenate(contours, axis=0)
    bx, by, bw, bh = cv2.boundingRect(all_pts)

    pad = max(8, int(min(h, w) * 0.03))
    x1  = max(0, bx - pad)
    y1  = max(0, by - pad)
    x2  = min(w, bx + bw + pad)
    y2  = min(h, by + bh + pad)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return (x1, y1, x2, y2)


def overlay_gradcam(image: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap (JET colourmap) over the original RGB image.

    Parameters
    ----------
    image   : (H, W, 3) uint8 RGB
    heatmap : (h_feat, w_feat) float in [0, 1]
    alpha   : overlay opacity

    Returns
    -------
    (H, W, 3) uint8 RGB blended image
    """
    h, w    = image.shape[:2]
    hmap_rs = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    hmap_u8      = (hmap_rs * 255).astype(np.uint8)
    hmap_colored = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    hmap_rgb     = cv2.cvtColor(hmap_colored, cv2.COLOR_BGR2RGB)

    blended = (image.astype(np.float32) * (1.0 - alpha)
               + hmap_rgb.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)
