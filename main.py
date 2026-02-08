import os
import argparse
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from dataset import SegDataset
from model import UNet, dice_loss_from_logits, tensor_to_image_np, tensor_to_mask_np


# =========================
# Config / Device utilities
# =========================
def load_config(path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 로드해서 dict로 반환"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg_device: str) -> torch.device:
    """
    device 선택 함수
    - "cpu" : 무조건 CPU
    - "cuda": CUDA 가능하면 GPU, 아니면 CPU
    - "auto": CUDA 가능하면 GPU, 아니면 CPU
    """
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(*dirs: str):
    """필요한 폴더가 없으면 생성"""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_output_dirs(cfg: Dict[str, Any]):
    """
    experiment.name 기반으로 출력 폴더를 통일해서 관리
      outputs/{exp_name}/checkpoints
      outputs/{exp_name}/predictions
    """
    exp_name = cfg["experiment"]["name"]
    ckpt_dir = f"outputs/{exp_name}/checkpoints"
    pred_dir = f"outputs/{exp_name}/predictions"
    return exp_name, ckpt_dir, pred_dir


# =========================
# Visualization (3-panel)
# =========================
def show_triplet(img_np, mask_np, title: str = "Sample"):
    """
    3분할 시각화:
      (a) Input Image
      (b) Overlay (Input + Mask)
      (c) Mask Only

    img_np  : (H,W,3) tensor or numpy, 0~1
    mask_np : (H,W) tensor or numpy, 0/1
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Input
    axes[0].imshow(img_np)
    axes[0].set_title("(a) Input")
    axes[0].axis("off")

    # (b) Overlay
    axes[1].imshow(img_np)
    axes[1].imshow(mask_np, alpha=0.4, cmap="Greens")  # 초록 overlay
    axes[1].set_title("(b) Overlay")
    axes[1].axis("off")

    # (c) Mask only
    axes[2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("(c) Mask")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def save_triplet(img_np, mask_np, save_path: str, title: str = ""):
    """
    3분할 이미지를 파일로 저장 (Input | Overlay | Mask)

    save_path 예:
      outputs/exp_01/predictions/0001_gt_triplet.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(mask_np, alpha=0.4, cmap="Greens")
    axes[1].set_title("Overlay")
    axes[1].axis("off")

    axes[2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Mask")
    axes[2].axis("off")

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def show_one_sample(dataset: SegDataset, index: int = 0, title_prefix: str = "GT"):
    """
    dataset에서 1장 꺼내서 3분할로 보여주기
    - Input
    - Overlay(GT)
    - Mask(GT)
    """
    img_t, mask_t = dataset[index]  # img: (C,H,W), mask: (1,H,W)
    img_np = tensor_to_image_np(img_t)   # (H,W,3)
    mask_np = tensor_to_mask_np(mask_t)  # (H,W)

    show_triplet(img_np, mask_np, title=f"{title_prefix} sample idx={index}")


# =========================
# Train
# =========================
def train(cfg: Dict[str, Any]):
    """
    학습 루프
    - loss = BCEWithLogitsLoss + DiceLoss
    - checkpoint 저장: outputs/{exp_name}/checkpoints/
    """
    device = get_device(cfg.get("device", "auto"))
    print(">>> Using device:", device)
    torch.manual_seed(int(cfg.get("seed", 42)))

    # 데이터 경로 / resize
    img_dir = cfg["data"]["img_dir"]
    mask_dir = cfg["data"]["mask_dir"]
    resize = cfg["data"].get("resize", None)
    if resize is not None:
        resize = (int(resize[0]), int(resize[1]))  # (H,W)

    ds = SegDataset(img_dir=img_dir, mask_dir=mask_dir, resize=resize)

    # train/val split
    val_ratio = float(cfg["train"].get("val_ratio", 0.2))
    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 모델 생성
    model = UNet(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),  # binary면 1
        base_channels=int(cfg["model"].get("base_channels", 64)),
    ).to(device)

    # optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    # binary segmentation에서 안정적으로 학습시키려고 BCE + Dice 같이 사용
    bce = nn.BCEWithLogitsLoss()

    # checkpoint dir (experiment 기반)
    _, ckpt_dir, _ = get_output_dirs(cfg)
    ensure_dirs(ckpt_dir)

    epochs = int(cfg["train"]["epochs"])

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0

        for img, mask in train_loader:
            img = img.to(device)     # (B,C,H,W)
            mask = mask.to(device)   # (B,1,H,W)

            logits = model(img)      # (B,1,H,W)

            # DiceLoss는 logits->sigmoid->dice 로 계산 (model.py에 정의됨)
            loss_dice = dice_loss_from_logits(logits, mask)
            # BCEWithLogitsLoss는 내부적으로 sigmoid 포함 (logits 바로 받음)
            loss_bce = bce(logits, mask)

            loss = loss_bce + loss_dice

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))

        # ---- validation ----
        model.eval()
        val_dice = 0.0
        n = 0

        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                mask = mask.to(device)

                logits = model(img)
                # dice score = 1 - dice_loss
                val_dice += float((1.0 - dice_loss_from_logits(logits, mask)).item())
                n += 1

        val_dice /= max(1, n)

        print(
            f"[Epoch {ep:d}/{epochs}] "
            f"train_loss={avg_loss:.4f}  "
            f"val_dice={val_dice:.4f}"
        )

        # ---- save checkpoint ----
        ckpt_path = os.path.join(ckpt_dir, f"unet_epoch{ep:02d}.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)

    print("Training done.")
    return ckpt_path  # 마지막 epoch 체크포인트 경로 반환


# =========================
# Inference (save images)
# =========================
@torch.no_grad()
def infer(cfg: Dict[str, Any], ckpt_path: str):
    """
    inference 실행 후, 결과를 '사진'으로 저장하는 파트
    - GT triplet 저장
    - Pred triplet 저장
    - 저장 위치: outputs/{exp_name}/predictions/
    """
    device = get_device(cfg.get("device", "auto"))

    # 데이터 로딩 (infer도 dataset에서 읽어옴)
    img_dir = cfg["data"]["img_dir"]
    mask_dir = cfg["data"]["mask_dir"]
    resize = cfg["data"].get("resize", None)
    if resize is not None:
        resize = (int(resize[0]), int(resize[1]))

    ds = SegDataset(img_dir=img_dir, mask_dir=mask_dir, resize=resize)

    # 모델 로딩
    model = UNet(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"].get("base_channels", 64)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 저장 폴더 (experiment 기반)
    _, _, pred_dir = get_output_dirs(cfg)
    ensure_dirs(pred_dir)

    thr = float(cfg["infer"].get("threshold", 0.5))
    num_samples = int(cfg["infer"].get("num_samples", 20))
    num_samples = min(num_samples, len(ds))

    print(f"Saving predictions to: {pred_dir}")
    print(f"threshold={thr}, num_samples={num_samples}")

    for i in range(num_samples):
        # 1장 로드
        img_t, gt_t = ds[i]                   # img: (C,H,W), gt: (1,H,W)
        img = img_t.unsqueeze(0).to(device)   # (1,C,H,W)

        # 모델 예측
        logits = model(img)
        probs = torch.sigmoid(logits)
        pred = (probs > thr).float()          # (1,1,H,W) 0/1

        # 시각화용 (CPU)
        img_np = tensor_to_image_np(img_t)      # (H,W,3)
        gt_np = tensor_to_mask_np(gt_t)         # (H,W)
        pred_np = tensor_to_mask_np(pred[0])    # (H,W)

        # ---- 3분할 저장 ----
        save_triplet(
            img_np,
            gt_np,
            save_path=os.path.join(pred_dir, f"{i:04d}_gt_triplet.png"),
            title=f"GT (idx={i})"
        )
        save_triplet(
            img_np,
            pred_np,
            save_path=os.path.join(pred_dir, f"{i:04d}_pred_triplet.png"),
            title=f"Pred (idx={i})"
        )

    print("Inference done.")


# =========================
# Main entry
# =========================
def main():
    """
    실행 예시:
      - dataset 확인 (GT 3분할):
          python main.py --mode show --index 0

      - 학습:
          python main.py --mode train

      - inference + 이미지 저장:
          python main.py --mode infer --ckpt outputs/exp_01/checkpoints/unet_epoch05.pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, choices=["show", "train", "infer", "all"], required=True)
    parser.add_argument("--index", type=int, default=0, help="for show mode")
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path for infer mode")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # output dirs 미리 생성 (experiment 기반)
    _, ckpt_dir, pred_dir = get_output_dirs(cfg)
    ensure_dirs(ckpt_dir, pred_dir)

    if args.mode == "show":
        # dataset에서 한 장 뽑아 3분할로 GT 확인
        resize = cfg["data"].get("resize", None)
        if resize is not None:
            resize = (int(resize[0]), int(resize[1]))

        ds = SegDataset(cfg["data"]["img_dir"], cfg["data"]["mask_dir"], resize=resize)
        show_one_sample(ds, index=args.index, title_prefix="GT")

    elif args.mode == "train":
        train(cfg)

    elif args.mode == "infer":
        if not args.ckpt:
            raise ValueError("infer 모드에서는 --ckpt 로 체크포인트 경로를 꼭 넣어줘야 해.")
        infer(cfg, ckpt_path=args.ckpt)

    elif args.mode == "all":
        # 1) show: 데이터/라벨 시각화 (index 사용)
        resize = cfg["data"].get("resize", None)
        if resize is not None:
            resize = (int(resize[0]), int(resize[1]))

        ds = SegDataset(cfg["data"]["img_dir"], cfg["data"]["mask_dir"], resize=resize)
        show_one_sample(ds, index=args.index, title_prefix="GT")

        # 2) train: 학습 돌리고 마지막 체크포인트 경로 받기
        last_ckpt_path = train(cfg)

        # 3) infer: 방금 학습한 체크포인트로 결과 저장
        infer(cfg, ckpt_path=last_ckpt_path)


if __name__ == "__main__":
    main()