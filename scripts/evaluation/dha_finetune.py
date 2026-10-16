"""
Cross-dataset adaptation on the Digital Hand Atlas (DHA) — fine-tuning (Sec. IV-F).

Starts from the RSNA-trained model and fine-tunes on a stratified 80/20 split of
DHA (stratified by ethnicity) with a reduced learning rate, so that the RSNA
representations are preserved while adapting to the target distribution.
The fine-tuned model is then evaluated on the held-out 20 % with the same
sex / ethnicity / age-group breakdown as the zero-shot study.
"""
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dha_zero_shot import (
    DEVICE, EMA, SEED, bone_age_loss, DHA_PATCH_ROOT, CHECKPOINT, DHADataset, load_dha, load_model, predict, subgroup_report,
)

OUT_DIR     = "./checkpoints/dha_finetune"
TEST_SIZE   = 0.2
BATCH_SIZE  = 16
NUM_WORKERS = 8
EPOCHS      = 20
LR          = 3e-5          # ~10x lower than RSNA training (3e-4)
EMA_DECAY   = 0.999


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_dha()
    ft_df, ho_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                                    stratify=df["ethnicity"])
    ft_df, ho_df = ft_df.reset_index(drop=True), ho_df.reset_index(drop=True)
    print(f"Fine-tune: {len(ft_df)}  |  Held-out: {len(ho_df)}")
    ft_df[["id"]].to_csv(os.path.join(OUT_DIR, "finetune_ids.csv"), index=False)
    ho_df[["id"]].to_csv(os.path.join(OUT_DIR, "heldout_ids.csv"), index=False)

    model = load_model(CHECKPOINT)
    loader = DataLoader(DHADataset(ft_df, DHA_PATCH_ROOT, train=True), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler() if DEVICE.startswith("cuda") else None
    ema       = EMA(model, decay=EMA_DECAY)

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Fine-tune epoch {epoch + 1}/{EPOCHS}")
        for img_h, img_r1, gender, age in pbar:
            img_h, img_r1 = img_h.to(DEVICE), img_r1.to(DEVICE)
            gender, age   = gender.to(DEVICE), age.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
                logits, _ = model(img_h, img_r1, gender)
                loss, _, _ = bone_age_loss(logits, age)

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
            ema.update(model)
            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        print(f"Epoch {epoch + 1}: train loss {running / len(loader):.4f}")

    ema.apply(model)
    model.eval()
    torch.save({"model_state_dict": model.state_dict(), "ema_state_dict": ema.shadow},
               os.path.join(OUT_DIR, "finetuned_model.pt"))

    out = predict(model, ho_df, DHA_PATCH_ROOT)
    out.to_csv(os.path.join(OUT_DIR, "heldout_predictions.csv"), index=False)

    print("\nFine-tuned DHA evaluation on held-out split (MAE, months, with TTA)")
    print(subgroup_report(out).to_string(index=False))


if __name__ == "__main__":
    main()
