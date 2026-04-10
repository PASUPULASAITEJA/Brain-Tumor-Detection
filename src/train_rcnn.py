"""
Faster RCNN Training — Brain Tumor Detection
=============================================
Fine-tunes a torchvision Faster RCNN (ResNet-50 + FPN backbone) on the
pseudo-annotated brain MRI dataset produced by generate_annotations.py.

Usage:
    python src/train_rcnn.py

Prerequisites:
    1. python src/augment_dataset.py
    2. python src/train_model.py
    3. python src/generate_annotations.py

Output:
    model/rcnn_model.pth
"""

import os
import sys
import json
import math
import time

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(0, os.path.dirname(__file__))
from rcnn_dataset import BrainTumorDetectionDataset

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

ANNOTATIONS_PATH = "annotations/annotations.json"
MODEL_SAVE_PATH  = "model/rcnn_model.pth"
NUM_CLASSES      = 2          # 0=background, 1=tumor
BATCH_SIZE       = 4          # larger batches = fewer iterations
EPOCHS           = 15
LR_HEAD          = 0.005
LR_BACKBONE      = 0.0005
WEIGHT_DECAY     = 5e-4
MOMENTUM         = 0.90
VAL_SPLIT        = 0.15
CONF_THRESHOLD   = 0.50       # threshold used in simple eval metric
MAX_TRAIN_IMAGES = 800        # more data breaks the convergence plateau

# Faster RCNN ops (NMS, ROI pooling) are not fully MPS-optimised and silently
# fall back to CPU, causing MPS↔CPU transfer overhead.  Pure CPU is faster here.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────────────────────────

def build_faster_rcnn(num_classes: int) -> torch.nn.Module:
    """
    Load COCO-pretrained Faster RCNN (MobileNetV3 Large 320 FPN backbone)
    and replace the box predictor head with a fresh head for `num_classes`.
    MobileNetV3 is ~3× faster than ResNet50 FPN and works well on Apple Silicon.
    """
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Collate (images are different sizes, so we return a list not a tensor)
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_detection_acc(model: torch.nn.Module,
                            loader: DataLoader,
                            device: torch.device,
                            threshold: float = CONF_THRESHOLD) -> float:
    """
    Simple binary accuracy: did the model correctly detect whether a
    tumour is present in each image?
    """
    model.eval()
    correct = total = 0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        outputs = model(imgs)
        for out, tgt in zip(outputs, targets):
            # Ground truth: image has a tumour box with label 1
            has_gt = any(l.item() == 1 for l in tgt["labels"])
            # Prediction: any box scored above threshold with label 1
            has_pred = any(
                s.item() >= threshold and l.item() == 1
                for s, l in zip(out["scores"], out["labels"])
            )
            correct += int(has_pred == has_gt)
            total   += 1
    model.train()
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train() -> None:
    os.makedirs("model", exist_ok=True)

    # ── Sanity checks ──────────────────────────────────────────────────────────
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"[ERROR] Annotations not found: '{ANNOTATIONS_PATH}'")
        print("        Run `python src/generate_annotations.py` first.")
        sys.exit(1)

    print(f"Device      : {DEVICE}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    full_ds = BrainTumorDetectionDataset(ANNOTATIONS_PATH)
    n_total = len(full_ds)
    print(f"\nAnnotated images : {n_total}")

    if n_total < 5:
        print("[ERROR] Too few annotated samples (need ≥5). "
              "Check that generate_annotations.py ran successfully.")
        sys.exit(1)

    # Cap total images so each epoch stays under ~2 min on CPU
    if n_total > MAX_TRAIN_IMAGES:
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
        full_ds = torch.utils.data.Subset(full_ds, indices[:MAX_TRAIN_IMAGES].tolist())
        n_total = MAX_TRAIN_IMAGES
        print(f"Capped to        : {n_total} images (set MAX_TRAIN_IMAGES to use more)")

    n_val   = max(1, int(VAL_SPLIT * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train / Val      : {n_train} / {n_val}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=2,          shuffle=False,
                              collate_fn=collate_fn, num_workers=2,
                              persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_faster_rcnn(NUM_CLASSES).to(DEVICE)

    # Different LR for backbone vs head
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ],
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # Cosine annealing with warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_acc = 0.0
    history      = []

    print(f"\nStarting training for {EPOCHS} epochs …\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Acc':>10}  {'Time':>8}")
    print("─" * 45)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss   = 0.0
        t0           = time.time()

        for imgs, targets in train_loader:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses    = sum(loss_dict.values())

            if not math.isfinite(losses.item()):
                print(f"  [!] Non-finite loss at epoch {epoch}, skipping batch.")
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += losses.item()

        scheduler.step()

        avg_loss = total_loss / max(1, len(train_loader))
        val_acc  = evaluate_detection_acc(model, val_loader, DEVICE)
        elapsed  = time.time() - t0

        flag = ""
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "num_classes":      NUM_CLASSES,
                },
                MODEL_SAVE_PATH,
            )
            flag = "  ✅ saved"

        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})
        print(f"{epoch:>6}  {avg_loss:>12.4f}  {val_acc*100:>9.1f}%  "
              f"{elapsed:>6.1f}s{flag}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"Training complete.")
    print(f"Best validation detection accuracy : {best_val_acc*100:.1f}%")
    print(f"Model saved → {MODEL_SAVE_PATH}")

    # Save training history
    history_path = "model/rcnn_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History   → {history_path}")
    print("\nRun `streamlit run app.py` to launch the web app.")


if __name__ == "__main__":
    train()
