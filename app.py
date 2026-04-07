"""
Brain Tumor Detection — Streamlit App
======================================
Detection pipeline (priority order):

  1. EfficientNetB0 binary classifier  →  tumour probability score
  2. Faster RCNN detector              →  bounding-box localisation (if model exists)
  3. Grad-CAM attention map            →  fallback localisation
  4. OpenCV morphological segmentation →  independent visual verification

CNN confidence tiers:
  ≥ 0.70  →  🔴  High confidence  — Tumour Detected
  ≥ 0.50  →  🟠  Moderate         — Possible Tumour
  ≥ 0.40  →  🟡  Borderline       — Inconclusive
  < 0.40  →  🟢  No Tumour Detected
"""

import os
import sys

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ── make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as _eff_preprocess

class EfficientNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return _eff_preprocess(x)
    def get_config(self):
        return super().get_config()

# Faster RCNN (optional — requires PyTorch + trained model)
try:
    from rcnn_detector import load_rcnn_model, detect_tumor, draw_detections
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# Grad-CAM (optional — requires trained EfficientNet model)
try:
    from gradcam import get_gradcam, heatmap_to_bbox, overlay_gradcam
    _GRADCAM_OK = True
except ImportError:
    _GRADCAM_OK = False

# ──────────────────────────────────────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brain Tumor Detection (RCNN)",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor Detection")
st.caption(
    "Powered by **EfficientNetB0 classifier** + **Faster RCNN detector** "
    "with Grad-CAM attention visualisation."
)

# ──────────────────────────────────────────────────────────────────────────────
#  Model paths
# ──────────────────────────────────────────────────────────────────────────────

CLASSIFIER_PATH  = "model/efficientnet_classifier.h5"
LEGACY_PATH      = "model/brain_tumor_model.h5"          # old MobileNetV2
RCNN_PATH        = "model/rcnn_model.pth"


# ──────────────────────────────────────────────────────────────────────────────
#  Cached model loaders
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading classifier…")
def _load_classifier() -> "tf.keras.Model | None":
    for path in (CLASSIFIER_PATH, LEGACY_PATH):
        if os.path.exists(path):
            return tf.keras.models.load_model(
                path, compile=False,
                custom_objects={"EfficientNetPreprocess": EfficientNetPreprocess}
            )
    return None


@st.cache_resource(show_spinner="Loading Faster RCNN…")
def _load_rcnn():
    if not _TORCH_OK:
        return None
    return load_rcnn_model(RCNN_PATH)


# ──────────────────────────────────────────────────────────────────────────────
#  Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
    """Return (1, size, size, 3) float32 array in [0, 255]."""
    img = image.convert("RGB").resize((size, size))
    return np.expand_dims(np.array(img, dtype=np.float32), 0)


# ──────────────────────────────────────────────────────────────────────────────
#  OpenCV morphological segmentation (fallback)
# ──────────────────────────────────────────────────────────────────────────────

MIN_COMPACTNESS = 0.40
MIN_INTENSITY   = 120


def _compute_threshold(blurred: np.ndarray) -> float:
    non_black = blurred[blurred > 10]
    if non_black.size == 0:
        return float(blurred.mean())
    t90 = float(np.percentile(non_black, 90))
    return float(blurred.mean() + 0.60 * blurred.std()) if t90 >= 250 else t90


def segment_tumor_opencv(image: Image.Image):
    """
    Pure-OpenCV morphological tumour segmentation (independent of CNN).

    Returns (result_rgb, found, info_dict)
    """
    img_rgb    = np.array(image.convert("RGB"))
    h, w       = img_rgb.shape[:2]
    image_area = h * w

    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh      = _compute_threshold(blurred)
    mask        = np.zeros((h, w), dtype=np.uint8)
    mask[blurred > thresh] = 255

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened  = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  k_open,  iterations=1)
    closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=3)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    if n_labels < 2:
        return img_rgb.copy(), False, {}

    border_margin = max(5, int(min(h, w) * 0.02))
    candidates    = []

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x0   = int(stats[i, cv2.CC_STAT_LEFT])
        y0   = int(stats[i, cv2.CC_STAT_TOP])
        bw_  = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_  = int(stats[i, cv2.CC_STAT_HEIGHT])

        if area < image_area * 0.003:  continue
        if area > image_area * 0.25:   continue
        if x0         <= border_margin:             continue
        if y0         <= border_margin:             continue
        if (x0 + bw_) >= (w - border_margin):       continue
        if (y0 + bh_) >= (h - border_margin):       continue

        blob_mask = (labels == i).astype(np.uint8)
        mean_int  = float(np.mean(gray[blob_mask > 0]))

        cnts, _ = cv2.findContours(blob_mask * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        compact = 0.0
        if cnts:
            perim = cv2.arcLength(cnts[0], True)
            if perim > 0:
                compact = (4.0 * np.pi * area) / (perim ** 2)

        if compact  < MIN_COMPACTNESS: continue
        if mean_int < MIN_INTENSITY:   continue

        area_pct     = area / image_area
        area_fitness = max(0.0, 1.0 - abs(area_pct - 0.05) / 0.10)
        score        = mean_int * 0.60 + compact * 100 * 0.25 + area_fitness * 100 * 0.15

        candidates.append(dict(
            mask=blob_mask, contours=cnts, score=score,
            area=area, intensity=mean_int, compact=compact,
            centroid=centroids[i],
        ))

    if not candidates:
        return img_rgb.copy(), False, {}

    best       = max(candidates, key=lambda c: c["score"])
    (cx, cy), _ = cv2.minEnclosingCircle(best["contours"][0])

    result  = img_rgb.copy()
    m3d     = np.stack([best["mask"]] * 3, axis=-1).astype(np.float32)
    red_lyr = np.zeros_like(result, dtype=np.float32)
    red_lyr[..., 0] = 255
    blended = result.astype(np.float32) * (1.0 - 0.50 * m3d) + red_lyr * (0.50 * m3d)
    result  = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.drawContours(result, best["contours"], -1, (255, 30, 30), 2)

    info = {
        "center":   (int(cx), int(cy)),
        "area_pct": round(best["area"] / image_area * 100, 2),
        "intensity": round(best["intensity"], 1),
        "compact":  round(best["compact"], 3),
    }
    return result, True, info


# ──────────────────────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model Status")
    clf_model  = _load_classifier()
    rcnn_model = _load_rcnn()

    st.write("**Classifier**",
             "✅ EfficientNetB0" if (clf_model and CLASSIFIER_PATH in str(type(clf_model)))
             else ("✅ MobileNetV2 (legacy)" if clf_model else "❌ Not found"))
    st.write("**Faster RCNN**",
             "✅ Ready" if rcnn_model else
             ("⚠️ PyTorch not installed" if not _TORCH_OK else "⚠️ Not trained yet"))
    st.write("**Grad-CAM**", "✅ Available" if _GRADCAM_OK else "⚠️ Unavailable")
    st.markdown("---")
    st.caption("Training pipeline:\n"
               "1. `python src/augment_dataset.py`\n"
               "2. `python src/train_model.py`\n"
               "3. `python src/generate_annotations.py`\n"
               "4. `python src/train_rcnn.py`")

uploaded = st.file_uploader(
    "Upload an MRI scan (JPG / PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("👆 Upload a brain MRI image to start detection.")
    st.stop()

try:
    image = Image.open(uploaded)
except Exception:
    st.error("❌ Could not open image file.")
    st.stop()

if clf_model is None:
    st.error("❌ No classifier model found. Run `python src/train_model.py` first.")
    st.stop()

# ── Layout ────────────────────────────────────────────────────────────────────
col_orig, col_rcnn, col_cam = st.columns(3)
with col_orig:
    st.image(image, caption="Original MRI", use_container_width=True)

with st.spinner("Analysing scan…"):

    arr  = preprocess(image)
    pred = float(clf_model.predict(arr, verbose=0)[0][0])

    # ── Confidence tier banner ─────────────────────────────────────────────────
    if pred >= 0.70:
        st.error(f"🔴 **Tumour Detected** — Classifier confidence: {pred*100:.1f}%")
    elif pred >= 0.50:
        st.warning(f"🟠 **Possible Tumour** — Confidence: {pred*100:.1f}%  "
                   f"_(Clinical review recommended)_")
    elif pred >= 0.40:
        st.info(f"🟡 **Borderline** — Confidence: {pred*100:.1f}%  "
                f"_(Inconclusive — consider follow-up scan)_")
    else:
        confidence_no = (1.0 - pred) * 100
        st.success(f"🟢 **No Tumour Detected** — Confidence: {confidence_no:.1f}%")

    # ── Faster RCNN detection ──────────────────────────────────────────────────
    if rcnn_model is not None and pred >= 0.40:
        result = detect_tumor(rcnn_model, image)
        rcnn_vis = draw_detections(image, result)
        with col_rcnn:
            st.image(rcnn_vis, caption="Faster RCNN Detection", use_container_width=True)
            if result["detected"]:
                st.caption(
                    f"📦 {len(result['boxes'])} region(s) found  |  "
                    f"Best score: {result['best_score']*100:.1f}%"
                )
            else:
                st.caption("RCNN: no high-confidence tumour region")
    elif pred >= 0.40:
        with col_rcnn:
            st.image(image, caption="RCNN not available", use_container_width=True)
            st.caption("Train the RCNN model to enable bounding-box detection.")

    # ── Grad-CAM / fallback visualisation ─────────────────────────────────────
    if _GRADCAM_OK:
        try:
            heatmap   = get_gradcam(clf_model, arr)
            cam_vis   = overlay_gradcam(np.array(image.convert("RGB")), heatmap)
            with col_cam:
                st.image(cam_vis, caption="Grad-CAM Attention Map",
                         use_container_width=True)

            # If RCNN not available and pred is positive, show bbox from GradCAM
            if (rcnn_model is None or not _TORCH_OK) and pred >= 0.50:
                bbox = heatmap_to_bbox(heatmap,
                                       (image.height, image.width),
                                       threshold=0.40)
                if bbox:
                    img_arr = np.array(image.convert("RGB"))
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(img_arr, (x1, y1), (x2, y2), (255, 60, 60), 2)
                    st.image(img_arr, caption="Grad-CAM Localisation",
                             use_container_width=True)
        except Exception:
            with col_cam:
                st.image(image, caption="Grad-CAM unavailable",
                         use_container_width=True)
    else:
        # OpenCV segmentation as pure fallback
        if pred >= 0.40:
            seg_vis, seg_found, seg_info = segment_tumor_opencv(image)
            with col_cam:
                st.image(seg_vis,
                         caption="OpenCV Segmentation" if seg_found
                         else "No distinct region found",
                         use_container_width=True)
                if seg_found:
                    cx_pct = seg_info["center"][0] / image.width  * 100
                    cy_pct = seg_info["center"][1] / image.height * 100
                    st.caption(
                        f"📍 Centre ≈ ({cx_pct:.0f}%, {cy_pct:.0f}%)  |  "
                        f"Area ≈ {seg_info['area_pct']}%  |  "
                        f"Brightness: {seg_info['intensity']}  |  "
                        f"Compactness: {seg_info['compact']}"
                    )

# ── Metrics panel ─────────────────────────────────────────────────────────────
metrics_path = "model/metrics.json"
if os.path.exists(metrics_path):
    import json
    with open(metrics_path) as f:
        m = json.load(f)
    with st.expander("📊 Classifier Training Metrics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{m.get('accuracy', 0)*100:.1f}%")
        c2.metric("AUC",       f"{m.get('auc', 0):.4f}")
        c3.metric("Precision", f"{m.get('precision', 0)*100:.1f}%")
        c4.metric("Recall",    f"{m.get('recall', 0)*100:.1f}%")
