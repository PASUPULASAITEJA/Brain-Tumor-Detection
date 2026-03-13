"""
Brain Tumor Detection — Streamlit App
======================================
Segmentation pipeline (pure OpenCV):

  1.  Grayscale + Gaussian blur (7×7)
  2.  DUAL-MODE adaptive threshold:
        • Normal MRI  → p90 of non-black pixels
        • Saturated   → mean + 0.60×std
  3.  Morphological OPENING  (3×3, 1 iter)  + CLOSING (9×9, 3 iters)
  4.  Connected-component analysis
  5.  Hard-reject: border-touching, <0.3%, >25% area
  6.  Quality filters  ← KEY FALSE-POSITIVE SUPPRESSOR
        • compactness  >= 0.40  (tumors are round; vessels/folds are elongated)
        • mean_intensity >= 120  (tumors are hyper-intense on T2/FLAIR)
  7.  Score: intensity (60%) + compactness (25%) + area-fitness (15%)
  8.  Semi-transparent red fill + 2-px contour outline

CNN confidence tiers:
  >= 0.70  →  🔴  High confidence — Tumor Detected
  >= 0.50  →  🟠  Moderate confidence — Possible Tumor
  >= 0.40  →  🟡  Low confidence — Inconclusive
  <  0.40  →  🟢  No Tumor Detected
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detection",
                   page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan to check whether a brain tumor is present.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model("model/brain_tumor_model.h5")


def preprocess_for_model(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  THRESHOLD HELPER
# ══════════════════════════════════════════════════════════════════════════════

def compute_threshold(blurred: np.ndarray) -> float:
    """
    Dual-mode adaptive threshold:
      Mode A (normal):    90th-percentile of non-black pixels
      Mode B (saturated): mean + 0.60×std  (when p90 ≥ 250)
    """
    non_black = blurred[blurred > 10]
    if non_black.size == 0:
        return float(blurred.mean())
    t90 = float(np.percentile(non_black, 90))
    if t90 >= 250:
        return float(blurred.mean() + 0.60 * blurred.std())
    return t90


# ══════════════════════════════════════════════════════════════════════════════
#  TUMOR SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

# Minimum quality thresholds — validated against real tumor and no-tumor scans:
#   Real tumors:      compactness 0.73–0.84,  mean brightness 196–221
#   False positives:  compactness 0.19–0.25,  mean brightness  89–110
MIN_COMPACTNESS = 0.40   # below this → elongated vessel / fold, not a tumor
MIN_INTENSITY   = 120    # below this → not hyper-intense enough to be a tumor


def segment_tumor(original_image: Image.Image):
    """
    Returns
    -------
    result_rgb : np.ndarray  — image with red overlay (or original if not found)
    found      : bool
    info       : dict
    """
    img_rgb    = np.array(original_image.convert("RGB"))
    h, w       = img_rgb.shape[:2]
    image_area = h * w

    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # ── Threshold ─────────────────────────────────────────────────────────────
    thresh      = compute_threshold(blurred)
    bright_mask = np.zeros((h, w), dtype=np.uint8)
    bright_mask[blurred > thresh] = 255

    # ── Morphological filtering ───────────────────────────────────────────────
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened  = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed  = cv2.morphologyEx(opened,     cv2.MORPH_CLOSE, k_close, iterations=3)

    # ── Connected-component analysis ──────────────────────────────────────────
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    if num_labels < 2:
        return img_rgb.copy(), False, {}

    # ── Candidate filtering ───────────────────────────────────────────────────
    border_margin = max(5, int(min(h, w) * 0.02))
    candidates    = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x0   = int(stats[i, cv2.CC_STAT_LEFT])
        y0   = int(stats[i, cv2.CC_STAT_TOP])
        bw   = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_  = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Hard area filter
        if area < image_area * 0.003:  continue
        if area > image_area * 0.25:   continue

        # Border filter (skull rim / artefacts)
        if x0         <= border_margin:       continue
        if y0         <= border_margin:       continue
        if (x0 + bw)  >= (w - border_margin): continue
        if (y0 + bh_) >= (h - border_margin): continue

        blob_mask = (labels == i).astype(np.uint8)
        mean_int  = float(np.mean(gray[blob_mask > 0]))

        blob_u8 = blob_mask * 255
        cnts, _ = cv2.findContours(blob_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        compact = 0.0
        if cnts:
            perim = cv2.arcLength(cnts[0], True)
            if perim > 0:
                compact = (4.0 * np.pi * area) / (perim ** 2)

        # ── Quality filter: suppresses false positives ────────────────────────
        # Vessels, cortical folds, and scan artefacts are elongated (low
        # compactness) and dimmer than true hyper-intense tumors.
        if compact  < MIN_COMPACTNESS:  continue
        if mean_int < MIN_INTENSITY:    continue

        area_pct     = area / image_area
        area_fitness = max(0.0, 1.0 - abs(area_pct - 0.05) / 0.10)
        score        = mean_int * 0.60 + compact * 100 * 0.25 + area_fitness * 100 * 0.15

        candidates.append({
            "mask":      blob_mask,
            "contours":  cnts,
            "score":     score,
            "area":      area,
            "intensity": mean_int,
            "compact":   compact,
            "centroid":  centroids[i],
        })

    if not candidates:
        return img_rgb.copy(), False, {}

    # ── Best candidate ────────────────────────────────────────────────────────
    best             = max(candidates, key=lambda c: c["score"])
    (cx, cy), radius = cv2.minEnclosingCircle(best["contours"][0])

    # ── Red overlay ───────────────────────────────────────────────────────────
    result  = img_rgb.copy()
    alpha   = 0.50
    m3d     = np.stack([best["mask"]] * 3, axis=-1).astype(np.float32)
    red_lyr = np.zeros_like(result, dtype=np.float32)
    red_lyr[..., 0] = 255
    blended = (result.astype(np.float32) * (1.0 - alpha * m3d)
               + red_lyr * (alpha * m3d))
    result  = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.drawContours(result, best["contours"], -1, (255, 30, 30), 2)

    info = {
        "center":    (int(cx), int(cy)),
        "radius":    int(radius),
        "area_pct":  round(best["area"] / image_area * 100, 2),
        "intensity": round(best["intensity"], 1),
        "compact":   round(best["compact"], 3),
    }
    return result, True, info


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "Upload MRI Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("❌ Invalid image file.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original MRI", width=300)

    with st.spinner("Analysing MRI scan…"):

        try:
            model = load_model()
        except Exception:
            st.error("❌ Model file not found. Please run train_model.py first.")
            st.stop()

        arr        = preprocess_for_model(image)
        prediction = float(model.predict(arr, verbose=0)[0][0])

        # ── Confidence-tiered response ─────────────────────────────────────────
        if prediction >= 0.40:

            overlay, seg_found, info = segment_tumor(image)

            # ── Tier 1: High confidence (≥70%) ────────────────────────────────
            if prediction >= 0.70:
                st.error(f"🔴 Tumor Detected  —  Confidence: {prediction*100:.1f}%")

            # ── Tier 2: Moderate confidence (50–70%) ──────────────────────────
            elif prediction >= 0.50:
                st.warning(f"🟠 Possible Tumor  —  Confidence: {prediction*100:.1f}%  "
                           f"_(Recommend clinical review)_")

            # ── Tier 3: Low / borderline (40–50%) ─────────────────────────────
            else:
                # If segmentation also found nothing → likely a false positive
                if not seg_found:
                    st.info(f"🟡 Inconclusive  —  Model confidence: {prediction*100:.1f}%  "
                            f"_(No distinct tumor region found — likely normal)_")
                else:
                    st.warning(f"🟠 Low-confidence detection  —  Confidence: {prediction*100:.1f}%  "
                               f"_(Please consult a radiologist)_")

            # ── Display segmentation result ────────────────────────────────────
            with col2:
                if seg_found:
                    st.image(overlay,
                             caption="Tumor Region Highlighted (Red)",
                             width=300)
                    cx_pct = info["center"][0] / image.width  * 100
                    cy_pct = info["center"][1] / image.height * 100
                    st.caption(
                        f"📍 Centre ≈ ({cx_pct:.0f}%, {cy_pct:.0f}%)  |  "
                        f"Area ≈ {info['area_pct']}%  |  "
                        f"Brightness: {info['intensity']}  |  "
                        f"Compactness: {info['compact']}"
                    )
                else:
                    st.image(image,
                             caption="No distinct tumor region isolated",
                             width=300)
                    if prediction >= 0.50:
                        st.caption(
                            "⚠️ Model flagged this scan but no compact bright "
                            "region was found. Tumor may be diffuse or isointense."
                        )

        else:
            # ── No tumor ──────────────────────────────────────────────────────
            confidence = 1.0 - prediction
            st.success(f"🟢 No Tumor Detected  —  Confidence: {confidence*100:.1f}%")
            with col2:
                st.image(image, caption="No Tumor Detected", width=300)

else:
    st.info("👆 Please upload an MRI image to start detection.")
