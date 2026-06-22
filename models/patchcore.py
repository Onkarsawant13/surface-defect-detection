"""
PatchCore Anomaly Detector
---------------------------
HOW IT WORKS (plain English):
  1. TRAIN  : extract patch features from all normal images → build a
              memory bank of "what normal looks like"
  2. TEST   : extract patch features from a new image → for each patch,
              find its nearest neighbour in the memory bank → anomaly
              score = max nearest-neighbour distance across all patches
  3. HEATMAP: upsample per-patch scores back to 224×224 → Gaussian
              smooth → you get a pixel-level anomaly map

WHY nearest-neighbour instead of a classifier?
  • No labels needed for defects — only normal images required
  • Generalises to unseen defect types automatically
  • Simple, interpretable, fast at inference

WHY coreset subsampling?
  Full memory bank = ~50k vectors for one category = slow at test time.
  Coreset picks the most representative ~10% subset using greedy
  farthest-point sampling — retains coverage, cuts search time 10×.

ALTERNATIVES to PatchCore:
  • PaDiM   : fits a Gaussian per patch position. Faster training,
              slightly lower AUROC. Good for simpler defect types.
  • FastFlow : normalising flow on features. State-of-art on some
              categories. Requires more GPU memory.
  • SimpleNet: lightweight classifier on top of features. Easiest to
              implement from scratch. ~2 pts below PatchCore on MVTec.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm

from models.feature_extractor import FeatureExtractor


class PatchCore:
    """
    Minimal PatchCore implementation.

    Usage:
        model = PatchCore(device="cuda")
        model.fit(train_loader)
        scores, masks = model.predict(test_loader)
    """

    def __init__(self, device: str = "cpu", coreset_ratio: float = 0.1,
                 neighbours: int = 9):
        self.device         = device
        self.coreset_ratio  = coreset_ratio   # keep 10 % of memory bank
        self.neighbours     = neighbours       # k in k-NN search
        self.extractor      = FeatureExtractor(device)
        self.memory_bank    = None             # set after fit()

    # ── public API ──────────────────────────────────────────────────────────

    def fit(self, train_loader):
        """Build memory bank from normal training images."""
        print("Building memory bank...")
        all_patches = []

        for batch in tqdm(train_loader, desc="Extracting train features"):
            features = self.extractor(batch["image"])
            patches  = self._pool_features(features)   # (B*N, D)
            all_patches.append(patches)

        all_patches = torch.cat(all_patches, dim=0)    # (total_patches, D)

        # coreset subsampling — keep representative subset
        self.memory_bank = self._coreset_subsample(all_patches)
        print(f"Memory bank: {self.memory_bank.shape[0]:,} patches "
              f"(D={self.memory_bank.shape[1]})")

    def predict(self, test_loader):
        """
        Returns:
            image_scores : list[float]  — one anomaly score per image
            anomaly_maps : list[np.array H×W] — pixel-level heatmaps
            labels       : list[int]    — ground truth (0=normal, 1=defective)
        """
        image_scores, anomaly_maps, labels = [], [], []

        for batch in tqdm(test_loader, desc="Scoring test images"):
            features   = self.extractor(batch["image"])
            patches    = self._pool_features(features)   # (N_patches, D)

            # nearest-neighbour distances
            distances  = self._knn_distance(patches)     # (N_patches,)

            # reshape to spatial map
            h = w = int(distances.shape[0] ** 0.5)
            score_map = distances.reshape(h, w).numpy()

            # upsample + smooth → 224×224 anomaly map
            score_map = self._upsample_smooth(score_map, size=224)

            image_scores.append(float(score_map.max()))
            anomaly_maps.append(score_map)
            labels.append(int(batch["label"].item()))

        return image_scores, anomaly_maps, labels

    # ── internals ───────────────────────────────────────────────────────────

    def _pool_features(self, features: dict) -> torch.Tensor:
        """
        Concatenate layer2 + layer3 features at matching resolution.
        layer3 is upsampled to match layer2 spatial dims before concat.
        Adaptive avg pool reduces each patch neighbourhood → one vector.
        """
        f2 = features["layer2"]   # (B, 512, 28, 28)
        f3 = features["layer3"]   # (B, 1024, 14, 14)

        # upsample f3 to 28×28
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                           align_corners=False)

        fused = torch.cat([f2, f3], dim=1)   # (B, 1536, 28, 28)

        # neighbourhood aggregation: 3×3 avg pool (stride=1, pad=1)
        fused = F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)

        B, C, H, W = fused.shape
        patches = fused.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return patches   # (B*H*W, 1536)

    def _coreset_subsample(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Greedy farthest-point coreset selection in a random-projection space.
        We project to 128-D first (Johnson-Lindenstrauss) to speed up distances.
        """
        n_select = max(1, int(len(patches) * self.coreset_ratio))
        projector = SparseRandomProjection(n_components=128, random_state=42)
        proj = projector.fit_transform(patches.numpy())
        proj = torch.tensor(proj, dtype=torch.float32)

        selected = [0]
        min_dists = torch.full((len(proj),), float("inf"))

        for _ in tqdm(range(n_select - 1), desc="Coreset sampling", leave=False):
            last = proj[selected[-1]].unsqueeze(0)
            dists = torch.norm(proj - last, dim=1)
            min_dists = torch.minimum(min_dists, dists)
            selected.append(int(min_dists.argmax()))

        return patches[selected]

    def _knn_distance(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient k-NN using ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        Avoids materialising the (N, M, D) difference tensor that caused OOM.
        chunk=32 keeps RAM under ~500 MB on CPU.
        """
        chunk     = 32
        bank      = self.memory_bank                       # (M, D)
        bank_norm = (bank ** 2).sum(dim=1, keepdim=True)   # (M, 1)
        all_dists = []

        for i in range(0, len(patches), chunk):
            p      = patches[i:i+chunk]                    # (c, D)
            p_norm = (p ** 2).sum(dim=1, keepdim=True)     # (c, 1)
            # (c, M)  —  no (c, M, D) tensor ever created
            dists_sq = (p_norm + bank_norm.T
                        - 2.0 * p @ bank.T).clamp(min=0)
            dists    = dists_sq.sqrt()
            topk     = dists.topk(self.neighbours, dim=1, largest=False)
            all_dists.append(topk.values.mean(dim=1))

        return torch.cat(all_dists)

    def _upsample_smooth(self, score_map: np.ndarray,
                         size: int = 224) -> np.ndarray:
        """Bilinear upsample to full image size + Gaussian smoothing."""
        t = torch.tensor(score_map).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                          align_corners=False)
        arr = t.squeeze().numpy()
        return gaussian_filter(arr, sigma=4)   # sigma=4 matches paper

    def predict_single(self, image_tensor) -> tuple:
        """
        Score a single image tensor without a DataLoader.
        Used by the Streamlit UI and FastAPI endpoint.

        Args:
            image_tensor : (3, 224, 224) normalised tensor

        Returns:
            (anomaly_score: float, heatmap: np.ndarray H x W)
        """
        features  = self.extractor(image_tensor.unsqueeze(0))
        patches   = self._pool_features(features)
        distances = self._knn_distance(patches)

        h = w = int(distances.shape[0] ** 0.5)
        score_map = distances.reshape(h, w).numpy()
        score_map = self._upsample_smooth(score_map, size=224)

        return float(score_map.max()), score_map

    def save(self, path: str):
        torch.save(self.memory_bank, path)
        print(f"Memory bank saved → {path}")

    def load(self, path: str):
        self.memory_bank = torch.load(path, map_location="cpu")
        print(f"Memory bank loaded ← {path} "
              f"({self.memory_bank.shape[0]:,} patches)")