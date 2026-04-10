"""
Faster RCNN Inference Module
=============================
Loads the trained Faster RCNN model and runs tumour detection on a PIL image.

Public API:
    load_rcnn_model(path)          -> nn.Module | None
    detect_tumor(model, image)     -> DetectionResult dict
    draw_detections(image, result) -> np.ndarray (RGB)
"""

import os
from typing import TypedDict

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TVF

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH  = "model/rcnn_model.pth"
DEFAULT_CONF_THRESH = 0.50
NUM_CLASSES         = 2        # background + tumor

if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
else:
    _DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
#  Types
# ──────────────────────────────────────────────────────────────────────────────

class DetectionResult(TypedDict):
    detected:   bool
    boxes:      list[list[float]]   # [[x1,y1,x2,y2], ...]
    scores:     list[float]
    best_score: float


# ──────────────────────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_rcnn_model(model_path: str = DEFAULT_MODEL_PATH,
                    device: torch.device = _DEVICE) -> "torch.nn.Module | None":
    """
    Load the trained Faster RCNN model from a checkpoint.
    Returns None if the file does not exist.
    """
    if not os.path.exists(model_path):
        return None

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def detect_tumor(model: "torch.nn.Module",
                 image: Image.Image,
                 threshold: float = DEFAULT_CONF_THRESH,
                 device: torch.device = _DEVICE) -> DetectionResult:
    """
    Run Faster RCNN detection on a PIL image.

    Parameters
    ----------
    model     : loaded Faster RCNN module
    image     : PIL.Image (any mode; will be converted to RGB)
    threshold : minimum confidence score to keep a detection
    device    : torch device

    Returns
    -------
    DetectionResult dict
    """
    img_tensor = TVF.to_tensor(image.convert("RGB")).to(device)  # (3,H,W)

    outputs = model([img_tensor])[0]

    boxes  = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    # Keep tumour-class detections (label == 1) above threshold
    keep = (scores >= threshold) & (labels == 1)
    boxes  = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return DetectionResult(detected=False, boxes=[], scores=[],
                               best_score=0.0)

    # Sort by score descending
    order  = np.argsort(scores)[::-1]
    boxes  = boxes[order]
    scores = scores[order]

    return DetectionResult(
        detected=True,
        boxes=boxes.tolist(),
        scores=scores.tolist(),
        best_score=float(scores[0]),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def draw_detections(image: Image.Image,
                    result: DetectionResult) -> np.ndarray:
    """
    Draw detection bounding boxes + confidence labels on the image.

    Returns a (H, W, 3) uint8 RGB numpy array.
    """
    img_rgb = np.array(image.convert("RGB"))

    if not result["detected"]:
        return img_rgb

    overlay = img_rgb.copy()

    for box, score in zip(result["boxes"], result["scores"]):
        x1, y1, x2, y2 = (int(v) for v in box)

        # Semi-transparent red fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 40, 40), -1)

    img_rgb = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)

    for box, score in zip(result["boxes"], result["scores"]):
        x1, y1, x2, y2 = (int(v) for v in box)

        # Solid red border (2 px)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (220, 40, 40), 2)

        # Label background + text
        label       = f"Tumor  {score*100:.1f}%"
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 0.55
        thickness   = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        lbl_y1      = max(y1 - th - 8, 0)
        cv2.rectangle(img_rgb,
                      (x1, lbl_y1), (x1 + tw + 6, lbl_y1 + th + 6),
                      (220, 40, 40), -1)
        cv2.putText(img_rgb, label,
                    (x1 + 3, lbl_y1 + th + 2),
                    font, font_scale, (255, 255, 255), thickness,
                    cv2.LINE_AA)

    return img_rgb
