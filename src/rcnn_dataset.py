"""
PyTorch Dataset for Faster RCNN
================================
Loads images + COCO-format bounding box annotations produced by
generate_annotations.py.

Each item returns:
    img_tensor : FloatTensor (3, H, W)  — pixel values in [0, 1]
    target     : dict with keys boxes, labels, area, iscrowd, image_id

Negative images (no tumour) have empty boxes/labels tensors, which is the
correct way to pass background samples to Faster RCNN.
"""

import json

import torch
from PIL import Image
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset


class BrainTumorDetectionDataset(Dataset):
    """
    Parameters
    ----------
    annotations_path : path to COCO annotations JSON
    transforms       : optional callable (img_tensor, target) -> (img_tensor, target)
    """

    def __init__(self, annotations_path: str, transforms=None):
        with open(annotations_path) as f:
            data = json.load(f)

        self.transforms = transforms

        # Index annotations by image_id
        ann_by_img: dict[int, list] = {}
        for ann in data["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        # Include ALL images — negatives have empty annotation lists so the
        # RCNN learns to suppress false positives on normal scans
        self.samples: list[tuple[dict, list[dict]]] = [
            (img_info, ann_by_img.get(img_info["id"], []))
            for img_info in data["images"]
        ]

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_info, anns = self.samples[idx]

        img = Image.open(img_info["file_name"]).convert("RGB")

        boxes: list[list[float]] = []
        labels: list[int]        = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x2, y2     = x + w, y + h

            # Clamp to image bounds
            x  = max(0.0, float(x))
            y  = max(0.0, float(y))
            x2 = min(float(img_info["width"]),  float(x2))
            y2 = min(float(img_info["height"]), float(y2))

            if x2 - x < 1 or y2 - y < 1:
                continue

            boxes.append([x, y, x2, y2])
            labels.append(int(ann["category_id"]))

        # Empty tensors are valid for Faster RCNN background images
        if boxes:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas    = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0,      dtype=torch.int64)
            areas    = torch.zeros(0,      dtype=torch.float32)

        iscrowd = torch.zeros(len(labels_t), dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "area":     areas,
            "iscrowd":  iscrowd,
            "image_id": torch.tensor([idx]),
        }

        img_tensor = TVF.to_tensor(img)   # (3, H, W) in [0, 1]

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target
