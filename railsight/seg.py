"""Rail-scene semantic segmentation (RailSem19, 19 classes).

All architectures share one interface so they are trained and compared with the same loop:
SegFormer B0/B1/B2 (transformers, Cityscapes-pretrained) and DeepLabV3 with a ResNet-50 or
MobileNetV3 backbone (CNN baselines, COCO-pretrained).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

RS19_CLASSES = [
    "road", "sidewalk", "construction", "tram-track", "fence", "pole", "traffic-light",
    "traffic-sign", "vegetation", "terrain", "sky", "human", "rail-track", "car", "truck",
    "trackbed", "on-rails", "rail-raised", "rail-embedded",
]  # fmt: skip
IGNORE_INDEX = 255
NUM_CLASSES = len(RS19_CLASSES)
CLS = {n: i for i, n in enumerate(RS19_CLASSES)}

# Pixels that make up a drivable track (used to find the ego path)
TRACK_IDS = [CLS["rail-track"], CLS["tram-track"], CLS["rail-raised"], CLS["rail-embedded"]]
# Surfaces that are normal to see on or between the rails; anything else there is foreign
GROUND_IDS = TRACK_IDS + [CLS[c] for c in ("trackbed", "terrain", "vegetation", "road", "sidewalk")]

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SEGFORMER_INIT = {
    f"segformer-{v}": f"nvidia/segformer-{v}-finetuned-cityscapes-1024-1024" for v in ("b0", "b1", "b2")
}
ARCHS = [*SEGFORMER_INIT, "deeplab-r50", "deeplab-mbv3"]

# Distinct colours for visualising the 19 classes (BGR)
PALETTE = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [0, 220, 220], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [128, 0, 128], [142, 0, 0], [70, 0, 0],
    [100, 100, 100], [0, 80, 100], [0, 0, 230], [32, 11, 119],
], np.uint8)  # fmt: skip


def load_rs19_class_names(config_json: Path) -> list[str]:
    with open(config_json, encoding="utf-8") as f:
        cfg = json.load(f)
    return [lab["name"] for lab in cfg["labels"]]


def normalize(rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 3xHxW float tensor."""
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


# ---------------------------------------------------------------- models
class SegFormerWrapper(nn.Module):
    def __init__(self, model, arch):
        super().__init__()
        self.model, self.arch = model, arch

    def forward(self, x):
        logits = self.model(pixel_values=x).logits  # 1/4 resolution
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def save(self, out_dir: Path):
        self.model.save_pretrained(out_dir)
        (Path(out_dir) / "arch.txt").write_text(self.arch)


class DeepLabWrapper(nn.Module):
    def __init__(self, model, arch):
        super().__init__()
        self.model, self.arch = model, arch

    def forward(self, x):
        return self.model(x)["out"]

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "deeplab.pt")
        (out_dir / "arch.txt").write_text(self.arch)


def build_model(arch: str, num_classes: int = NUM_CLASSES, weights: str | None = None) -> nn.Module:
    """Create a model for training (weights=None) or load trained weights from a run folder."""
    if arch.startswith("segformer"):
        from transformers import SegformerForSemanticSegmentation

        m = SegformerForSemanticSegmentation.from_pretrained(
            weights or SEGFORMER_INIT[arch],
            num_labels=num_classes,
            id2label=dict(enumerate(RS19_CLASSES)),
            label2id=CLS,
            ignore_mismatched_sizes=True,
        )
        return SegFormerWrapper(m, arch)
    if arch.startswith("deeplab"):
        from torchvision.models import segmentation as tvseg

        ctor, w = {
            "deeplab-r50": (tvseg.deeplabv3_resnet50, tvseg.DeepLabV3_ResNet50_Weights.DEFAULT),
            "deeplab-mbv3": (tvseg.deeplabv3_mobilenet_v3_large,
                             tvseg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT),
        }[arch]
        if weights:
            m = ctor(weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False)
            m.load_state_dict(torch.load(Path(weights) / "deeplab.pt", map_location="cpu"))
        else:
            m = ctor(weights=w)
            m.classifier[4] = nn.Conv2d(256, num_classes, 1)
            m.aux_classifier = None
        return DeepLabWrapper(m, arch)
    raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")


def detect_arch(weights_dir: str | Path) -> str:
    return (Path(weights_dir) / "arch.txt").read_text().strip()


# ---------------------------------------------------------------- inference
class Segmenter:
    """Loads a trained model and returns per-pixel labels + confidence for a BGR frame."""

    def __init__(self, weights_dir: str | Path, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = build_model(detect_arch(weights_dir), weights=str(weights_dir))
        self.net.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        # pad to a multiple of 32 so both architectures accept any working resolution
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        x = normalize(rgb)[None].to(self.device)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            logits = self.net(x)[0, :, :h, :w].float()
        prob = logits.softmax(0)
        conf, labels = prob.max(0)
        return labels.cpu().numpy().astype(np.uint8), conf.cpu().numpy()


def colorize(labels: np.ndarray) -> np.ndarray:
    out = PALETTE[np.clip(labels, 0, NUM_CLASSES - 1)]
    out[labels == IGNORE_INDEX] = 0
    return out
