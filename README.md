This is the project developed for Bone Age detection of RSNA dataset ( status: submitted at a conf.)
# Dual-Region Attention Fusion for Bone Age Estimation

Bone age estimation from hand radiographs using a shared-weight two-stream EfficientNetV2-S encoder, a learned region-attention fusion module, and a CORAL ordinal regression head. Submitted to INDICON 2026.

**MAE: 5.825 months** (RSNA, TTA) · 21.7M params · 54.3 ms/patient inference

## Architecture

![Architecture](architecture/architecture.png)

Two regions per radiograph — the full hand and a finger-joint crop — are encoded by a shared backbone, fused via a learned attention weighting, conditioned on gender, and passed through a 227-threshold CORAL head.

## Repo layout

```
architecture/   diagram source (.drawio) + exported image
src/            training scripts including ablations
  train.py            
  train_coral.py      
  train_modified.py   
samples/        example radiographs (RSNA, DHA)
```

## Results

| Dataset | Setting | MAE (months) |
|---|---|---|
| RSNA (held-out) | Standard | 5.921 |
| RSNA (held-out) | + TTA | **5.825** |
| DHA | Zero-shot | 11.61 |
| DHA | Fine-tuned | 10.26 |

## Citation

See link for the full manuscript. Citation details to be added upon publication.
