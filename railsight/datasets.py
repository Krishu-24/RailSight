import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .seg import IGNORE_INDEX, normalize


class RailSem19(Dataset):
    """Pre-resized RailSem19 produced by scripts/prepare_railsem19.py."""

    def __init__(self, root: str | Path, split: str, crop: int = 512, train: bool = True):
        self.root = Path(root)
        with open(self.root / "splits.json", encoding="utf-8") as f:
            self.ids = json.load(f)[split]
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.ids)

    def _load(self, i):
        name = self.ids[i]
        img = cv2.imread(str(self.root / "images" / f"{name}.jpg"))[:, :, ::-1]
        mask = cv2.imread(str(self.root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, i):
        img, mask = self._load(i)
        if self.train:
            img, mask = self._augment(img, mask)
        else:
            img, mask = _pad_to_multiple(img, mask, 32)
        return normalize(np.ascontiguousarray(img)), torch.from_numpy(mask.astype(np.int64))

    def _augment(self, img, mask):
        rng = np.random
        s = rng.uniform(0.75, 1.5)
        h, w = mask.shape
        nh, nw = int(h * s), int(w * s)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        c = self.crop
        if nh < c or nw < c:  # pad with "ignore" so the loss skips it
            ph, pw = max(c - nh, 0), max(c - nw, 0)
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
            nh, nw = mask.shape
        y, x = rng.randint(0, nh - c + 1), rng.randint(0, nw - c + 1)
        img, mask = img[y : y + c, x : x + c], mask[y : y + c, x : x + c]
        if rng.rand() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        # photometric jitter: brightness / contrast
        a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return img, np.ascontiguousarray(mask)


def _pad_to_multiple(img, mask, m):
    h, w = mask.shape
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    return img, mask
