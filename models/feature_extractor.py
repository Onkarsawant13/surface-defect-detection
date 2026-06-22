"""
Feature Extractor — WideResNet-50
----------------------------------
WHY WideResNet-50?
  The original PatchCore paper (Roth et al., 2022) benchmarks several
  backbones and finds WideResNet-50-2 achieves the best AUROC on MVTec AD
  (~99.1% image-level). Key reasons:
    • Wider channels → richer mid-level features per spatial location
    • Layers 2 & 3 (stride 8/16) give the best spatial resolution vs
      semantics trade-off for anomaly localisation
    • Pretrained on ImageNet — no fine-tuning needed (anomaly detection
      is unsupervised; we only use normal images)

WHY layers 2 & 3 specifically?
  Layer 1 = too low-level (edges, colours)
  Layer 4 = too semantic (object identity, loses spatial detail)
  Layers 2+3 = mid-level textures + structures — perfect for defects

ALTERNATIVES:
  • ResNet-18/50      : lighter, ~1–2 pt AUROC drop, good if GPU is limited
  • EfficientNet-B4   : slightly better accuracy, slower inference
  • ViT-B/16 (timm)  : transformer backbone, strong on textured surfaces,
                        requires timm library (`pip install timm`)
  • DINOv2 (Facebook) : self-supervised, excellent zero-shot features,
                        no ImageNet label dependency — cutting-edge option
"""

import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class FeatureExtractor(nn.Module):
    """
    Extracts multi-scale feature maps from layers 2 and 3 of WideResNet-50.
    Both hooks run in a single forward pass — no double inference cost.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

        # load pretrained weights (ImageNet1K)
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval()

        # freeze — we NEVER update backbone weights
        for p in backbone.parameters():
            p.requires_grad = False

        # keep only the feature layers we need
        self.layer1 = backbone.layer1   # stride 4  — not used but kept for ref
        self.layer2 = backbone.layer2   # stride 8  → 28×28 feature map
        self.layer3 = backbone.layer3   # stride 16 → 14×14 feature map
        self.prefix = nn.Sequential(    # stem: conv1 + bn + relu + maxpool
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.to(device)

        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward hooks to capture intermediate outputs."""
        def hook(name):
            def _hook(module, input, output):
                self._features[name] = output
            return _hook

        self.layer2.register_forward_hook(hook("layer2"))
        self.layer3.register_forward_hook(hook("layer3"))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x : (B, 3, 224, 224) normalised image batch

        Returns:
            dict with keys 'layer2' → (B, 512, 28, 28)
                           'layer3' → (B, 1024, 14, 14)
        """
        self._features.clear()
        x = x.to(self.device)
        x = self.prefix(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return {k: v.cpu() for k, v in self._features.items()}
