"""
Efficiency and deployability profiling (Table IV).

Measures, for the dual-region model at 448 x 448:
  * total parameters
  * computational complexity (FLOPs) of one standard forward pass
  * per-patient inference latency at batch size 1, standard and with flip TTA
"""
from torch.utils.flop_counter import FlopCounterMode
import time
import torch

import torch.nn as nn
import torch.nn.functional as F


SEED = 42
torch.manual_seed(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

IMG_SIZE    = 448
NUM_BINS    = 228


# BoneAgeModel — two-stream attention
class BoneAgeModel(nn.Module):
    """
    Two-stream EfficientNetV2-S with per-region softmax attention.

    Backbone is shared (weight-tied) across both streams.
    Two separate forward passes ensure each region produces its own
    distinct feature vector — the prerequisite for attention to work.
    """

    def __init__(self):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=True, num_classes=0
        )

        # Discover feature dimension
        with torch.no_grad():
            dummy    = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feat_dim = self.backbone(dummy).shape[1]  # 1280 for EffNetV2-S

        self.feat_dim = feat_dim

        # Attention gate: projects each region's feature vector to a
        # scalar score, then softmax across the two regions.
        self.attn_proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # scalar score per region
        )

        # Prediction head
        self.fc1      = nn.Linear(feat_dim + 1, 768)
        self.bn1      = nn.BatchNorm1d(768)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2      = nn.Linear(768, 256)
        self.bn2      = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc_coral = nn.Linear(256, NUM_BINS - 1)

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """One 3-ch image -> pooled feature vector [B, D]."""
        return self.backbone(img)

    def forward(self, img_hand: torch.Tensor, img_region1: torch.Tensor, gender: torch.Tensor):
        # 1. Two separate backbone passes -> two distinct feature vectors
        f_h  = self._encode(img_hand)     # [B, D]
        f_r1 = self._encode(img_region1)  # [B, D]

        # 2. Stack -> [B, 2, D]
        feats = torch.stack([f_h, f_r1], dim=1)

        # 3. Score each region independently
        B, R, D = feats.shape
        scores  = self.attn_proj(feats.view(B * R, D)).view(B, R)  # [B, 2]
        weights = F.softmax(scores, dim=1)                          # [B, 2]

        # 4. Weighted sum -> fused [B, D]
        fused = (weights.unsqueeze(-1) * feats).sum(dim=1)

        # 5. Predict
        g      = gender.unsqueeze(1)  # [B, 1]
        x      = F.relu(self.bn1(self.fc1(torch.cat([fused, g], dim=1))))
        x      = self.dropout1(x)
        x      = F.relu(self.bn2(self.fc2(x)))
        x      = self.dropout2(x)
        logits = self.fc_coral(x)     # [B, 227]

        return logits, weights  # always return weights so caller can log


CHECKPOINT = None        # optional: path to a trained checkpoint
WARMUP     = 50
ITERS      = 200


@torch.no_grad()
def time_per_patient(fn, warmup=WARMUP, iters=ITERS) -> float:
    for _ in range(warmup):
        fn()
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0   # ms


def main():
    model = BoneAgeModel().to(DEVICE).eval()
    if CHECKPOINT:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    img_h  = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    img_r1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    gender = torch.ones(1, device=DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(img_h, img_r1, gender)
    flops = counter.get_total_flops()

    def standard():
        model(img_h, img_r1, gender)

    def tta():
        model(img_h, img_r1, gender)
        model(torch.flip(img_h, [3]), torch.flip(img_r1, [3]), gender)

    t_std = time_per_patient(standard)
    t_tta = time_per_patient(tta)

    device_name = torch.cuda.get_device_name() if DEVICE.startswith("cuda") else "CPU"
    print(f"Device                    : {device_name}")
    print(f"Total parameters          : {n_params / 1e6:.1f} M")
    print(f"Computational complexity  : {flops / 1e9:.1f} GFLOPs")
    print(f"Standard inference latency: {t_std:.1f} ms/patient ({1000 / t_std:.1f} patients/s)")
    print(f"TTA inference latency     : {t_tta:.1f} ms/patient ({1000 / t_tta:.1f} patients/s)")


if __name__ == "__main__":
    main()