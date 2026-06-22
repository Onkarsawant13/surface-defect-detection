# Surface Defect Detection — PatchCore on MVTec AD

## Project structure
```
surface_defect/
├── data/
│   └── dataset.py          # MVTec AD loader + transforms
├── models/
│   ├── feature_extractor.py  # WideResNet-50 backbone, layers 2+3
│   └── patchcore.py          # Memory bank, coreset sampling, k-NN scoring
├── utils/
│   └── metrics.py            # AUROC, F1, heatmap + distribution plots
├── train_eval.py             # Entry point
└── requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt

python train_eval.py \
  --data   /path/to/mvtec_anomaly_detection \
  --category bottle \
  --output  outputs/
```

## How it works
1. **Preprocessing** — images resized to 224×224, normalised with ImageNet stats
2. **Feature extraction** — WideResNet-50 layers 2 & 3 (no fine-tuning)
3. **Memory bank** — all normal patch features stored, then coreset-subsampled to 10%
4. **Scoring** — each test patch scored by distance to its k nearest neighbours
5. **Heatmap** — per-patch scores upsampled + Gaussian smoothed to 224×224

## Expected results (bottle category)
| Metric       | Score  |
|-------------|--------|
| Image AUROC  | ~98.5% |
| Pixel AUROC  | ~96.0% |

## Backbone alternatives
| Backbone         | Image AUROC | Speed   | Notes                       |
|-----------------|-------------|---------|----------------------------|
| WideResNet-50   | ~99.1%      | Medium  | Paper default, recommended |
| ResNet-18       | ~97.5%      | Fast    | Good if GPU limited        |
| EfficientNet-B4 | ~98.8%      | Medium  | via `timm`                 |
| ViT-B/16        | ~98.5%      | Slower  | Transformer, cutting-edge  |
| DINOv2-Base     | ~99.3%      | Slower  | Self-supervised, best      |
