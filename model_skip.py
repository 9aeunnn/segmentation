import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1-2) Leaky ReLU
# -------------------------
def double_conv(in_ch: int, out_ch: int, use_bn: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.1, inplace=True))

    layers += [
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.1, inplace=True))

    return nn.Sequential(*layers)


# -------------------------
# 2) Down: MaxPool -> DoubleConv
# -------------------------
class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = double_conv(in_ch, out_ch, use_bn)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


# -------------------------
# 3) Up (NO SKIP):
#    Upsample -> DoubleConv
# -------------------------
class UpNoSkip(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = double_conv(in_ch, out_ch, use_bn)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


# -------------------------
# 4) UNet WITHOUT skip connection
# -------------------------
class UNet(nn.Module):
    """
    No-Skip UNet (Encoder-Decoder CNN)

    입력 : (B, in_channels, H, W)
    출력 : (B, out_channels, H, W)  # logits
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        use_bn: bool = True,
    ):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc   = double_conv(in_channels, c, use_bn)
        self.down1 = Down(c, c * 2, use_bn)
        self.down2 = Down(c * 2, c * 4, use_bn)
        self.down3 = Down(c * 4, c * 8, use_bn)
        self.down4 = Down(c * 8, c * 16, use_bn)

        # Decoder (NO SKIP → 채널 수 주의!)
        self.up1 = UpNoSkip(c * 16, c * 8, use_bn)
        self.up2 = UpNoSkip(c * 8,  c * 4, use_bn)
        self.up3 = UpNoSkip(c * 4,  c * 2, use_bn)
        self.up4 = UpNoSkip(c * 2,  c,     use_bn)

        # Output
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # c
        x2 = self.down1(x1)   # 2c
        x3 = self.down2(x2)   # 4c
        x4 = self.down3(x3)   # 8c
        x5 = self.down4(x4)   # 16c

        # Decoder (NO SKIP)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        logits = self.outc(x)
        return logits


# -------------------------
# 5) Dice Loss (logits 입력)
# -------------------------
def dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.contiguous()
    target = target.contiguous()

    inter = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


# -------------------------
# 6) 시각화 보조
# -------------------------
@torch.no_grad()
def tensor_to_image_np(img_t: torch.Tensor) -> torch.Tensor:
    x = img_t.detach().cpu().clamp(0, 1)
    x = x.permute(1, 2, 0).contiguous()
    return x


@torch.no_grad()
def tensor_to_mask_np(mask_t: torch.Tensor) -> torch.Tensor:
    x = mask_t.detach().cpu()
    if x.ndim == 3:
        x = x[0]
    return x.float().clamp(0, 1)
