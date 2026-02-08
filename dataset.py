import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(folder: str) -> List[Path]:
    p = Path(folder)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def _find_mask_path(mask_dir: str, stem: str) -> Optional[Path]:
    """mask_dir 안에서 stem(파일명) 동일한 마스크를 확장자 상관없이 찾기"""
    p = Path(mask_dir)
    for ext in IMG_EXTS:
        candidate = p / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # 혹시 대소문자/다른 확장자 섞였을 때 대비: 전체 탐색
    for f in p.iterdir():
        if f.is_file() and f.stem == stem:
            return f
    return None


class SegDataset(Dataset):
    """
    반환:
      image: FloatTensor (C,H,W), 0~1
      mask : FloatTensor (1,H,W), 0 or 1  (binary)
    """

    def __init__(self, img_dir: str, mask_dir: str, resize: Optional[Tuple[int, int]] = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.resize = resize  # (H,W) or None

        self.img_paths = _list_images(img_dir)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in: {img_dir}")

        # 이미지-마스크 매칭
        self.pairs = []
        for img_path in self.img_paths:
            mask_path = _find_mask_path(mask_dir, img_path.stem)
            if mask_path is None:
                raise RuntimeError(f"Mask not found for image: {img_path.name} (stem={img_path.stem})")
            self.pairs.append((img_path, mask_path))

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if self.resize is not None:
            # 이미지: bilinear
            img = img.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)
        return img

    def _load_mask(self, path: Path) -> Image.Image:
        mask = Image.open(path).convert("L")  # grayscale
        if self.resize is not None:
            # 마스크: nearest (중요! 값 번짐 방지)
            mask = mask.resize((self.resize[1], self.resize[0]), resample=Image.NEAREST)
        return mask

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        img_np = np.array(img, dtype=np.float32) / 255.0  # (H,W,3)
        mask_np = np.array(mask, dtype=np.float32)        # (H,W), 0~255 or 0~1

        # binary로 강제: 0/255면 >0 으로 1 처리
        mask_np = (mask_np > 0).astype(np.float32)

        # torch tensor (C,H,W)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # (3,H,W)
        mask_t = torch.from_numpy(mask_np)[None, ...].contiguous()      # (1,H,W)

        return img_t, mask_t
