# Brain Tumor Detection — Faster RCNN + EfficientNetB0

A deep-learning system for detecting and **localising** brain tumours in MRI scans.
The pipeline combines a fine-tuned **EfficientNetB0 binary classifier** with a
**Faster RCNN (MobileNetV3 Large 320 FPN)** object detector to produce both a
confidence score and a bounding box around the tumour region, displayed in an
interactive Streamlit web application.

---

## Results

| Model | Metric | Value |
|---|---|---|
| EfficientNetB0 Classifier | Validation AUC | **99.75%** |
| EfficientNetB0 Classifier | Validation Accuracy | **~97.5%** |
| Faster RCNN Detector | Detection Accuracy | **93.3%** |

*Detection accuracy = binary correct/incorrect for each image: did the model correctly
detect presence or absence of a tumour?*

---

## Architecture Overview

```
MRI Image
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 — EfficientNetB0 Classifier                │
│                                                     │
│  Input (224×224×3)                                  │
│    → RandomFlip / Rotation / Zoom / Contrast        │
│    → EfficientNetPreprocess (custom Keras layer)    │
│    → EfficientNetB0 backbone (ImageNet, frozen)     │
│    → GlobalAveragePooling2D                         │
│    → BatchNorm → Dropout(0.4)                       │
│    → Dense(256, relu) → Dropout(0.3)               │
│    → Dense(1, sigmoid)                              │
│                                                     │
│  Output: tumour probability in [0, 1]               │
└───────────────────┬─────────────────────────────────┘
                    │ if prob ≥ 0.40
                    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 — Faster RCNN Detector                     │
│                                                     │
│  Backbone  : MobileNetV3 Large 320 FPN              │
│  Pretrain  : COCO weights                           │
│  Head      : FastRCNNPredictor (2 classes)          │
│  Training  : Fine-tuned on GradCAM pseudo-labels    │
│                                                     │
│  Output: bounding boxes + confidence scores         │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3 — Grad-CAM Attention Map                   │
│                                                     │
│  Splits model into pre_model / backbone / head      │
│  Uses GradientTape on backbone output activations   │
│  Produces class activation heatmap overlaid on MRI  │
│                                                     │
│  Also used to auto-generate RCNN training labels    │
└─────────────────────────────────────────────────────┘
```

---

## Dataset

| Split | Class | Count |
|---|---|---|
| Original | Tumour (yes) | 155 |
| Original | Normal (no) | 98 |
| **Original Total** | | **253** |
| After augmentation | Tumour (yes) | 1,705 |
| After augmentation | Normal (no) | 1,078 |
| **Augmented Total** | | **2,783** |

Augmentation techniques applied (10× expansion):
- Horizontal flip
- Vertical flip
- 90° rotation
- 270° rotation
- Brightness increase (+30%)
- Brightness decrease (−30%)
- Gaussian noise
- +15° rotation
- −15° rotation
- Zoom in (80% crop)

---

## Project Structure

```
brain-tumor-detection/
├── dataset/                        # Original MRI images (253 total)
│   ├── yes/                        #   155 tumour scans
│   └── no/                         #   98 normal scans
│
├── dataset_augmented/              # 10× augmented dataset (~2,783 images)
│   ├── yes/                        #   1,705 tumour scans
│   └── no/                         #   1,078 normal scans
│
├── annotations/
│   └── annotations.json            # COCO-format bounding boxes (auto-generated)
│
├── model/
│   ├── efficientnet_classifier.h5  # Fine-tuned EfficientNetB0 (~20 MB)
│   ├── rcnn_model.pth              # Fine-tuned Faster RCNN (~72 MB)
│   ├── metrics.json                # Classifier training metrics
│   └── rcnn_history.json           # RCNN loss + val_acc per epoch
│
├── src/
│   ├── augment_dataset.py          # Step 1 — 10× data augmentation
│   ├── train_model.py              # Step 2 — EfficientNetB0 training
│   ├── gradcam.py                  # Grad-CAM computation + heatmap→bbox
│   ├── generate_annotations.py     # Step 3 — GradCAM → COCO annotations
│   ├── rcnn_dataset.py             # PyTorch Dataset (COCO format)
│   ├── train_rcnn.py               # Step 4 — Faster RCNN fine-tuning
│   └── rcnn_detector.py            # RCNN inference + visualisation
│
├── app.py                          # Streamlit web application
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- TensorFlow 2.16+, numpy 1.26.4
- PyTorch ≥ 2.1, torchvision ≥ 0.16
- OpenCV, Pillow, scikit-learn, Streamlit, matplotlib

> **Note (Apple Silicon / conda):** Install packages into the active conda environment
> using `python -m pip install` to avoid environment mismatch issues.

---

## Training Pipeline

Run the following scripts **in order**:

---

### Step 1 — Augment the Dataset

```bash
python src/augment_dataset.py
```

Reads from `dataset/` and writes 10 augmented variants of each image into
`dataset_augmented/`. Expands 253 → ~2,783 images.

---

### Step 2 — Train the EfficientNetB0 Classifier

```bash
python src/train_model.py
```

**Architecture:**
- Frozen EfficientNetB0 backbone (ImageNet weights)
- Custom `EfficientNetPreprocess` Keras layer (required for Keras 3 H5 serialisation)
- Data augmentation layers: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation
- Classification head: GAP → BatchNorm → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(1, sigmoid)

**Callbacks:** EarlyStopping (val_auc, patience=6), ModelCheckpoint, ReduceLROnPlateau

**Output:** `model/efficientnet_classifier.h5`, `model/metrics.json`

**Achieved:** val_auc = 0.9975, val_accuracy ≈ 97.5%

---

### Step 3 — Generate Bounding Box Annotations

```bash
python src/generate_annotations.py
```

For each tumour image where the classifier predicts ≥ 0.50 confidence:
1. Runs Grad-CAM on the image
2. Thresholds the heatmap at 0.40 to find the activated region
3. Converts that region to a COCO-format bounding box

Also includes all normal (no-tumour) images as negative samples with no boxes,
so the RCNN learns to suppress detections on healthy scans.

**Output:** `annotations/annotations.json`
- Tumour images with boxes: **1,396**
- Normal images (no boxes): **1,078**
- Total: **2,474 images**

---

### Step 4 — Fine-tune Faster RCNN

```bash
python src/train_rcnn.py
```

**Model:** `fasterrcnn_mobilenet_v3_large_320_fpn` (COCO pretrained)
- Box predictor head replaced with `FastRCNNPredictor(in_features, num_classes=2)`
- Backbone LR: 0.0005, Head LR: 0.005
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- Scheduler: CosineAnnealingLR
- Gradient clipping: max_norm=1.0
- Trained on 800 images (balanced positive/negative), 15 epochs

**Training Results:**

| Epoch | Train Loss | Val Accuracy |
|---|---|---|
| 1 | 0.698 | 88.3% |
| 5 | 0.814 | 92.5% |
| 12 | 0.677 | 93.3% |
| 15 | 0.675 | **93.3%** |

**Output:** `model/rcnn_model.pth`, `model/rcnn_history.json`

---

### Step 5 — Launch the Web App

```bash
streamlit run app.py
```

---

## Web Application

The Streamlit app displays three panels side-by-side for each uploaded MRI:

| Panel | Description |
|---|---|
| **Original MRI** | Uploaded scan as-is |
| **Faster RCNN** | Bounding box drawn over the tumour region with confidence score |
| **Grad-CAM** | Class activation heatmap overlaid on the scan |

The sidebar shows live model status (classifier + RCNN loaded or not) and a
metrics panel with accuracy, AUC, and loss from the last training run.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| EfficientNetB0 over original MobileNetV2 | Higher accuracy at similar inference speed |
| Classifier head only — no backbone fine-tuning | Phase 1 alone achieves val-AUC 0.9975; full fine-tuning risked overfitting and caused model corruption |
| Custom `EfficientNetPreprocess` Keras layer | Keras 3 cannot deserialise `Lambda` layers from H5 — a proper subclass with `get_config()` is required |
| GradCAM pre_model / backbone / head split | Keras 3 `GradientTape` cannot trace nested Functional sub-models; splitting avoids the tracing limitation |
| GradCAM pseudo-annotations for RCNN | Eliminates need for manual bounding-box labelling on 1,700+ images |
| Negative images included in RCNN dataset | Without them, val_acc is trivially 100% (all GT = positive); negatives make the metric meaningful |
| MobileNetV3 Large 320 FPN over ResNet-50 FPN | ~3× faster per epoch on CPU; fits MacBook memory; still COCO pretrained |
| CPU over MPS for RCNN | Faster RCNN NMS and ROI pooling ops silently fall back from MPS to CPU, causing transfer overhead — pure CPU is faster |
| 10× augmentation | Overcomes the small original dataset (253 → ~2,783 images) |
