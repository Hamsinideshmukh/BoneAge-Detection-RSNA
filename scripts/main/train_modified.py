# Modifications
# 1- fixed val set held out
# 2- EARLY_STOP_MIN_DELTA loosened: 0.01 -> 0.003
# 3- LR scheduler patience: 6 -> 5, and EARLY_STOP_PATIENCE set to 5x that (25), so folds get more room to actually converge.
import os
import time
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = "/home/cs24d0008/BoneAge"

CSV_FILENAME = "/home/cs24d0008/BoneAge/boneage-training-dataset.csv"

CHECKPOINT_DIR = "/home/cs24d0008/BoneAge/checkpoints2"

IMG_EXT = ".jpg"

GPU_INDEX = 0

CSV_PATH   = os.path.join(DATA_ROOT, CSV_FILENAME)
PATCH_ROOT = os.path.join(DATA_ROOT, "patches")  

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# configuration
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_INDEX)
    DEVICE = f"cuda:{GPU_INDEX}"
else:
    DEVICE = "cpu"

IMG_SIZE    = 448
NUM_BINS    = 228
BATCH_SIZE  = 24          # A100 40GB can comfortably handle this two-stream setup at 448px
NUM_WORKERS = 8

N_FOLDS = 5               

VAL_SIZE = 500

MAX_EPOCHS_PER_FOLD = 80   # hard ceiling; early stopping will normally trigger first

LR_PATIENCE = 5
EARLY_STOP_PATIENCE = LR_PATIENCE * 5   # 25

# loosened min-delta so small genuine gains still reset patience
EARLY_STOP_MIN_DELTA = 0.003

LR_INIT = 3e-4

CORAL_WEIGHT = 0.6
HUBER_WEIGHT = 1.0

USE_EMA   = True
EMA_DECAY = 0.999

USE_TTA = True

# Patient-leakage check
def check_patient_leakage(df: pd.DataFrame) -> bool:
    n_images   = len(df)
    n_patients = df["id"].nunique()

    if n_images == n_patients:
        print(
            f"Patient-leakage check PASSED: {n_images} images, "
            f"{n_patients} unique IDs — one image per patient."
        )
        return False  # no leakage
    else:
        print(
            f"Leakage detected: {n_images} images, {n_patients} patients. "
            f"(Note: fixed-size 500-sample holdout per fold does not "
            f"group by patient — verify this is acceptable for your data.)"
        )
        return True

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

# Dataset
# Filenames are the image "id" from the CSV, e.g. patches/hand/1234.jpg
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_region(folder: str, img_id: int) -> np.ndarray:
    path = os.path.join(PATCH_ROOT, folder, f"{img_id}{IMG_EXT}")
    img  = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return img  # HxWx3 uint8


def _to_tensor(img: np.ndarray, flip: bool, alpha: float, beta: float) -> torch.Tensor:
    if flip:
        img = np.fliplr(img).copy()
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255)
    t   = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (t - _NORM_MEAN) / _NORM_STD  # [3, H, W]


class BoneAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True):
        self.df    = df.reset_index(drop=True)
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row    = self.df.iloc[idx]
        img_id = int(row.id)

        hand_np = _load_region("hand",    img_id)
        r1_np   = _load_region("region1", img_id)

        # Shared augmentation — both regions get identical spatial transform
        flip  = self.train and (np.random.rand() > 0.5)
        alpha = np.random.uniform(0.9, 1.1) if self.train else 1.0
        beta  = np.random.uniform(-5.0, 5.0) if self.train else 0.0

        img_hand    = _to_tensor(hand_np, flip, alpha, beta)  # [3, H, W]
        img_region1 = _to_tensor(r1_np,   flip, alpha, beta)  # [3, H, W]

        gender = torch.tensor(1.0 if row.male else -1.0, dtype=torch.float32)
        age    = torch.clamp(torch.tensor(row.boneage, dtype=torch.float32), 1, 228)

        return img_hand, img_region1, gender, age


# BoneAgeModel — two-stream attention
class BoneAgeModel(nn.Module):
    """
    Two-stream EfficientNetV2-S with real per-region softmax attention.

    Backbone is shared (weight-tied) across both streams.
    Two separate forward passes ensure each region produces its own
    distinct feature vector — the prerequisite for attention to work.
    """

    def __init__(self):
        super().__init__()
        import timm

        # Standard 3-channel backbone — no channel-stacking hack
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
    l_mae      = (wts * torch.abs(age_pred - age_true)).mean()
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


# Train a single fold
def train_one_fold(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    fold_dir = os.path.join(CHECKPOINT_DIR, f"fold_{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}  |  Train: {len(train_df)}  Val: {len(val_df)}")
    print("=" * 70)

    train_ds = BoneAgeDataset(train_df, train=True)
    val_ds   = BoneAgeDataset(val_df,   train=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model     = BoneAgeModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler() if DEVICE.startswith("cuda") else None
    ema       = EMA(model, decay=EMA_DECAY) if USE_EMA else None

    # Plateau-based LR schedule — used instead of the original fixed-epoch
    # drops because with early stopping each fold may run a different
    # number of epochs, so a fixed epoch schedule doesn't transfer cleanly
    # across folds.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=LR_PATIENCE, min_lr=1e-6
    )

    best_mae         = float("inf")
    epochs_no_improve = 0
    best_ckpt_path    = os.path.join(fold_dir, "best_model.pt")

    for epoch in range(MAX_EPOCHS_PER_FOLD):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]["lr"]

        # train
        model.train()
        train_loss = train_mae = train_coral = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"[Fold {fold_idx + 1}] Epoch {epoch + 1}/{MAX_EPOCHS_PER_FOLD} [LR={current_lr:.2e}]",
        )
        for img_h, img_r1, gender, age in pbar:
            img_h  = img_h.to(DEVICE,  non_blocking=True)
            img_r1 = img_r1.to(DEVICE, non_blocking=True)
            gender = gender.to(DEVICE, non_blocking=True)
            age    = age.to(DEVICE,    non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
                logits, _ = model(img_h, img_r1, gender)
                loss, mae, coral = bone_age_loss(logits, age)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            if ema is not None:
                ema.update(model)

            train_loss  += loss.item()
            train_mae   += mae.item()
            train_coral += coral.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "mae": f"{mae.item():.3f}"})

        n = len(train_loader)
        train_loss /= n
        train_mae  /= n
        train_coral /= n

        # val
        model.eval()
        if ema is not None:
            ema.apply(model)

        val_mae = val_mae_tta = 0.0
        all_w_hand, all_w_r1 = [], []

        with torch.no_grad():
            for img_h, img_r1, gender, age in val_loader:
                img_h  = img_h.to(DEVICE,  non_blocking=True)
                img_r1 = img_r1.to(DEVICE, non_blocking=True)
                gender = gender.to(DEVICE, non_blocking=True)
                age    = age.to(DEVICE,    non_blocking=True)

                logits, weights = model(img_h, img_r1, gender)
                pred_age        = predict_age_coral(logits)
                val_mae        += torch.abs(pred_age - age).mean().item()

                all_w_hand.extend(weights[:, 0].cpu().tolist())
                all_w_r1.extend(weights[:, 1].cpu().tolist())

                if USE_TTA:
                    pred_tta     = predict_with_tta(model, img_h, img_r1, gender)
                    val_mae_tta += torch.abs(pred_tta - age).mean().item()

        val_mae     /= len(val_loader)
        val_mae_tta /= len(val_loader) if USE_TTA else 1

        if ema is not None:
            ema.restore(model)

        w_h  = torch.tensor(all_w_hand)
        w_r1 = torch.tensor(all_w_r1)

        selection_mae = val_mae_tta if USE_TTA else val_mae
        scheduler.step(selection_mae)

        epoch_time = time.time() - epoch_start
        print(f"\nFold {fold_idx + 1} | Epoch {epoch + 1}/{MAX_EPOCHS_PER_FOLD}  ({epoch_time:.1f}s)")
        print(f"  Train — Loss: {train_loss:.4f}, MAE: {train_mae:.3f}, CORAL: {train_coral:.4f}")
        print(f"  Val   — MAE: {val_mae:.3f}" + (f", TTA: {val_mae_tta:.3f}" if USE_TTA else ""))
        print(
            f"  Attn  — Hand: {w_h.mean():.3f} +/- {w_h.std():.3f} "
            f"[{w_h.min():.3f}, {w_h.max():.3f}]  |  "
            f"R1: {w_r1.mean():.3f} +/- {w_r1.std():.3f} "
            f"[{w_r1.min():.3f}, {w_r1.max():.3f}]"
            f"  (std>0 confirms weights vary)"
        )

        # ckpt and early stopping
        if selection_mae < best_mae - EARLY_STOP_MIN_DELTA:
            best_mae = selection_mae
            epochs_no_improve = 0
            save_dict = {
                "fold":                 fold_idx + 1,
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae":              val_mae,
                "val_mae_tta":          val_mae_tta if USE_TTA else None,
                "train_loss":           train_loss,
                "train_mae":            train_mae,
            }
            if ema is not None:
                save_dict["ema_state_dict"] = ema.shadow
            torch.save(save_dict, best_ckpt_path)
            print(f"  NEW BEST for fold {fold_idx + 1} — {best_mae:.3f}" + (" (TTA)" if USE_TTA else "") + f"  -> saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs (best so far: {best_mae:.3f})")

        print()

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f">>> Early stopping fold {fold_idx + 1} at epoch {epoch + 1} (no improvement for {EARLY_STOP_PATIENCE} epochs)\n")
            break

    print(f"Fold {fold_idx + 1} complete. Best val MAE: {best_mae:.3f}  |  checkpoint: {best_ckpt_path}")
    return {"fold": fold_idx + 1, "best_mae": best_mae, "checkpoint": best_ckpt_path}


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(GPU_INDEX)}")

    print(f"\nReading labels CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")

    check_patient_leakage(df)

    print("Filtering to samples with both hand and region1 patches present...")
    valid_mask = [
        os.path.exists(os.path.join(PATCH_ROOT, "hand",    f"{int(r.id)}{IMG_EXT}")) and
        os.path.exists(os.path.join(PATCH_ROOT, "region1", f"{int(r.id)}{IMG_EXT}"))
        for _, r in df.iterrows()
    ]
    df = df[valid_mask].reset_index(drop=True)
    print(f"Valid samples after filtering: {len(df)}")

    fold_splits = []
    for fold_idx in range(N_FOLDS):
        val_df_fold = df.sample(n=VAL_SIZE, random_state=SEED + fold_idx)
        val_idx     = val_df_fold.index.to_numpy()
        train_idx   = df.drop(val_df_fold.index).index.to_numpy()
        fold_splits.append((train_idx, val_idx))

    print("\n" + "#" * 70)
    print(f"# STARTING {N_FOLDS}-FOLD CROSS VALIDATION")
    print(f"# Fixed validation holdout per fold: {VAL_SIZE} images")
    print(f"# Max epochs/fold: {MAX_EPOCHS_PER_FOLD}  |  LR patience: {LR_PATIENCE}  |  Early-stop patience: {EARLY_STOP_PATIENCE}")
    print(f"# Checkpoints will be saved under: {CHECKPOINT_DIR}")
    print("#" * 70)

    fold_results = []
    overall_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        result = train_one_fold(fold_idx, train_df, val_df)
        fold_results.append(result)

    total_time = time.time() - overall_start

    # final
    maes = [r["best_mae"] for r in fold_results]
    summary_lines = ["=" * 70, "5-FOLD CROSS VALIDATION COMPLETE", "=" * 70]
    for r in fold_results:
        summary_lines.append(f"Fold {r['fold']}: best val MAE = {r['best_mae']:.3f}  ({r['checkpoint']})")
    summary_lines.append("-" * 70)
    summary_lines.append(f"Mean MAE across folds: {np.mean(maes):.3f}  (+/- {np.std(maes):.3f})")
    summary_lines.append(f"Total wall time: {total_time / 3600:.2f} hours")
    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(CHECKPOINT_DIR, "cv_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
