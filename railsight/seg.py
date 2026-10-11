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

from .classes import (  # noqa: F401  (re-exported for existing imports)
    CLS,
    GROUND_IDS,
    IGNORE_INDEX,
    NUM_CLASSES,
    RAIL_IDS,
    RS19_CLASSES,
    TRACK_IDS,
)

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


def build_model(arch: str, num_classes: int = NUM_CLASSES, weights: str | None = None,
                attn: str | None = None) -> nn.Module:
    """Create a model for training (weights=None) or load trained weights from a run folder.
    `attn="sdpa"` selects PyTorch's fused attention kernels for SegFormer (faster inference)."""
    if arch.startswith("segformer"):
        from transformers import SegformerForSemanticSegmentation

        kwargs = {"attn_implementation": attn} if attn else {}
        m = SegformerForSemanticSegmentation.from_pretrained(
            weights or SEGFORMER_INIT[arch],
            num_labels=num_classes,
            id2label=dict(enumerate(RS19_CLASSES)),
            label2id=CLS,
            ignore_mismatched_sizes=True,
            **kwargs,
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
PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


class Segmenter:
    """Loads a trained model and returns per-pixel labels + confidence for a BGR frame.

    `precision` sets autocast on the GPU. Measured on an RTX 5060 Laptop at 960x544: SegFormer-B2
    is fastest in bf16 with fused attention (36 vs 49 ms per frame with eager fp16), while the
    tiny B0 is fastest in plain fp32 because autocast overhead outweighs its compute.
    """

    def __init__(self, weights_dir: str | Path, device: str = "cuda", precision: str = "bf16"):
        self.device = device if torch.cuda.is_available() else "cpu"
        arch = detect_arch(weights_dir)
        attn = "sdpa" if arch.startswith("segformer") else None
        self.net = build_model(arch, weights=str(weights_dir), attn=attn)
        self.net.eval().to(self.device)
        self.dtype = PRECISIONS[precision] if self.device == "cuda" else None
        self.mean = torch.tensor(MEAN * 255, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD * 255, device=self.device).view(1, 3, 1, 1)
        torch.backends.cudnn.benchmark = True  # fixed input size: let cuDNN pick the fastest kernels

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        # pad to a multiple of 32 so every architecture accepts any working resolution
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        bgr = cv2.copyMakeBorder(frame_bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        # upload raw bytes and normalise on the GPU (much cheaper than float maths on the CPU)
        x = torch.from_numpy(bgr).to(self.device).permute(2, 0, 1)[None].flip(1).float()
        x = (x - self.mean) / self.std
        with torch.autocast(self.device, dtype=self.dtype or torch.float32, enabled=self.dtype is not None):
            logits = self.net(x)[0, :, :h, :w].float()
        prob = logits.softmax(0)
        conf, labels = prob.max(0)
        return labels.cpu().numpy().astype(np.uint8), conf.cpu().numpy()


def colorize(labels: np.ndarray) -> np.ndarray:
    out = PALETTE[np.clip(labels, 0, NUM_CLASSES - 1)]
    out[labels == IGNORE_INDEX] = 0
    return out
