"""
Cross-dataset evaluation on the Digital Hand Atlas (DHA) — zero-shot (Sec. IV-F).

The RSNA-trained model is applied to DHA without any adaptation. Reports overall
MAE and the breakdown by sex, ethnicity and age group (Tables V and VI), and
writes per-image predictions.

DHA labels CSV columns: id, boneage (months), male (0/1), ethnicity.
Hand / region-1 patches are extracted with extract_regions.py, as for RSNA.
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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


CORAL_WEIGHT = 0.6
HUBER_WEIGHT = 1.0


# EMA
class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)

# CORAL helpers
def coral_targets(age: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Vectorized CORAL target creation.
    age : [B] float — cast to long internally so comparison is exact.
    Returns [B, num_classes-1] binary matrix.
    """
    age_long   = age.long()
    thresholds = torch.arange(1, num_classes, device=age.device).view(1, -1)
    return (age_long.view(-1, 1) > thresholds).float()


def coral_loss(logits: torch.Tensor, age_true: torch.Tensor) -> torch.Tensor:
    levels = coral_targets(age_true, NUM_BINS)
    return F.binary_cross_entropy_with_logits(logits, levels, reduction="mean")


def predict_age_coral(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).sum(dim=1) + 1

_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tensor(img: np.ndarray, flip: bool, alpha: float, beta: float) -> torch.Tensor:
    if flip:
        img = np.fliplr(img).copy()
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255)
    t   = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (t - _NORM_MEAN) / _NORM_STD  # [3, H, W]


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

# loss
def bone_age_loss(logits: torch.Tensor, age_true: torch.Tensor):
    age_pred   = predict_age_coral(logits)
    loss_coral = coral_loss(logits, age_true)
    loss_huber = F.smooth_l1_loss(age_pred, age_true, reduction="mean")
    wts        = 1.0 + 0.5 * (age_true < 36).float() + 0.3 * (age_true > 150).float()
    l_mae      = (wts * torch.abs(age_pred - age_true)).mean()   # logged only
    total_loss = CORAL_WEIGHT * loss_coral + HUBER_WEIGHT * loss_huber
    return total_loss, l_mae, loss_coral


# TTA
def predict_with_tta(model: nn.Module, img_h: torch.Tensor, img_r1: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
    """Average of original + horizontally-flipped prediction."""
    with torch.no_grad():
        logits_orig, _ = model(img_h, img_r1, gender)
        p_orig         = predict_age_coral(logits_orig)

        logits_flip, _ = model(
            torch.flip(img_h,  [3]),
            torch.flip(img_r1, [3]),
            gender,
        )
        p_flip = predict_age_coral(logits_flip)

    return (p_orig + p_flip) / 2.0


DHA_CSV        = "./data/dha/dha_labels.csv"
DHA_PATCH_ROOT = "./data/dha/patches"            # hand/ and region1/
CHECKPOINT     = "./checkpoints/main/fold_1/best_model.pt"
OUT_CSV        = "./results/dha_zero_shot_predictions.csv"
IMG_EXT        = ".jpg"
BATCH_SIZE     = 32
NUM_WORKERS    = 8

AGE_BINS   = [0, 36, 72, 132, np.inf]            # months
AGE_LABELS = ["0-3 yr", "3-6 yr", "6-11 yr", "11+ yr"]


class DHADataset(Dataset):
    def __init__(self, df: pd.DataFrame, patch_root: str, train: bool = False):
        self.df, self.patch_root, self.train = df.reset_index(drop=True), patch_root, train

    def __len__(self):
        return len(self.df)

    def _load(self, folder, img_id):
        img = cv2.imread(os.path.join(self.patch_root, folder, f"{img_id}{IMG_EXT}"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img_id = int(row.id)
        flip   = self.train and (np.random.rand() > 0.5)
        alpha  = np.random.uniform(0.9, 1.1) if self.train else 1.0
        beta   = np.random.uniform(-5.0, 5.0) if self.train else 0.0

        img_h  = _to_tensor(self._load("hand",    img_id), flip, alpha, beta)
        img_r1 = _to_tensor(self._load("region1", img_id), flip, alpha, beta)
        gender = torch.tensor(1.0 if row.male else -1.0, dtype=torch.float32)
        age    = torch.clamp(torch.tensor(float(row.boneage), dtype=torch.float32), 1, 228)
        return img_h, img_r1, gender, age


def load_dha(csv_path=DHA_CSV, patch_root=DHA_PATCH_ROOT) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    keep = [
        os.path.exists(os.path.join(patch_root, "hand",    f"{int(i)}{IMG_EXT}")) and
        os.path.exists(os.path.join(patch_root, "region1", f"{int(i)}{IMG_EXT}"))
        for i in df["id"]
    ]
    return df[keep].reset_index(drop=True)


def load_model(checkpoint: str) -> BoneAgeModel:
    """Builds BoneAgeModel and loads the EMA weights from a train_modified.py checkpoint."""
    model = BoneAgeModel().to(DEVICE)
    ckpt  = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    ref   = model.state_dict()
    model.load_state_dict({k: v.to(ref[k].dtype) for k, v in state.items()})
    return model.eval()


@torch.no_grad()
def predict(model, df: pd.DataFrame, patch_root: str) -> pd.DataFrame:
    loader = DataLoader(DHADataset(df, patch_root), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    preds, preds_tta = [], []
    for img_h, img_r1, gender, _ in tqdm(loader, desc="DHA inference"):
        img_h, img_r1, gender = img_h.to(DEVICE), img_r1.to(DEVICE), gender.to(DEVICE)
        logits, _ = model(img_h, img_r1, gender)
        preds.append(predict_age_coral(logits).cpu())
        preds_tta.append(predict_with_tta(model, img_h, img_r1, gender).cpu())

    out = df.copy()
    out["pred"]     = torch.cat(preds).numpy()
    out["pred_tta"] = torch.cat(preds_tta).numpy()
    out["abs_err"]  = (out["pred_tta"] - out["boneage"]).abs()
    return out


def subgroup_report(out: pd.DataFrame) -> pd.DataFrame:
    out = out.assign(
        sex=np.where(out["male"].astype(bool), "Male", "Female"),
        age_group=pd.cut(out["boneage"], AGE_BINS, labels=AGE_LABELS, right=False),
    )
    rows = [{"group": "Overall", "subgroup": "All", "n": len(out), "MAE": out["abs_err"].mean()}]
    for col, name in [("sex", "Sex"), ("ethnicity", "Ethnicity"), ("age_group", "Age group")]:
        for key, g in out.groupby(col, observed=True):
            rows.append({"group": name, "subgroup": key, "n": len(g), "MAE": g["abs_err"].mean()})
    return pd.DataFrame(rows).round({"MAE": 2})


def main():
    df = load_dha()
    print(f"DHA images with patches: {len(df)}")

    model = load_model(CHECKPOINT)
    out   = predict(model, df, DHA_PATCH_ROOT)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("\nZero-shot DHA evaluation (MAE, months, with TTA)")
    print(subgroup_report(out).to_string(index=False))


if __name__ == "__main__":
    main()