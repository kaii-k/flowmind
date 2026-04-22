"""
flow_encoder.py
================
Neural encoder for context-aware flow embeddings.

Architecture:
  FlowEncoder      — 3-layer MLP that maps raw flow features → 64-dim embedding
  ContrastiveLoss  — NT-Xent (SimCLR-style) loss to pull same-class flows together
  FlowClassifier   — kNN classifier trained on top of frozen embeddings
  EmbeddingMetrics — Cosine similarity analysis (intra-class vs inter-class)

Usage (standalone training):
  python flow_encoder.py --train --data embeddings.csv

Usage (from main.py):
  from flow_encoder import FlowEncoder, train_encoder, FlowClassifier, EmbeddingMetrics
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class FlowEncoder(nn.Module):
    """
    Context-aware flow embedding encoder.

    Maps a flow feature vector (bytes, packets, timing, protocol flags…)
    to a 64-dimensional L2-normalised embedding space where:
      - Similar traffic types (e.g. YouTube + Netflix) cluster together
      - Dissimilar types (streaming vs gaming) are pushed apart

    Architecture: 3-layer MLP with BatchNorm + residual projection
    Input:  [batch, input_dim]  — raw normalised flow features
    Output: [batch, 64]         — L2-normalised embedding vector
    """

    def __init__(self, input_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, embed_dim),
        )

        # Residual projection so gradients flow cleanly
        self.residual_proj = nn.Linear(input_dim, embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x) + self.residual_proj(x)
        # L2-normalise so cosine similarity == dot product
        return F.normalize(out, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONTRASTIVE LOSS  (NT-Xent / SimCLR style)
# ══════════════════════════════════════════════════════════════════════════════

class ContrastiveLoss(nn.Module):
    """
    Supervised Triplet Margin Loss with cosine distance.

    For each anchor in the batch:
      - Finds the hardest positive (same class, lowest cosine sim)
      - Finds the hardest negative (different class, highest cosine sim)
      - Applies margin: loss = max(0, d_pos - d_neg + margin)

    Why Triplet instead of NT-Xent:
      - NT-Xent overflows in FP16 when temperature < 0.1 with large batches
      - Triplet loss uses cosine distance directly, immune to scale issues
      - Works correctly with highly imbalanced class distributions

    Margin=0.3 means embeddings of different classes must be at least
    0.3 cosine distance units further apart than same-class embeddings.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        n = embeddings.size(0)
        if n < 2:
            return embeddings.sum() * 0.0

        # Cast to float32 for numerical stability (safe even in FP16 training)
        emb = embeddings.float()

        # Cosine distance matrix = 1 - cosine_similarity
        # embeddings are already L2-normalised so dot product = cosine sim
        sim  = torch.mm(emb, emb.T).clamp(-1, 1)
        dist = 1.0 - sim   # cosine distance [0, 2]

        labels_r = labels.view(-1, 1)
        pos_mask = (labels_r == labels_r.T)
        neg_mask = ~pos_mask

        # Mask diagonal
        eye = torch.eye(n, dtype=torch.bool, device=emb.device)
        pos_mask = pos_mask & ~eye
        neg_mask = neg_mask & ~eye

        losses = []
        for i in range(n):
            if not pos_mask[i].any() or not neg_mask[i].any():
                continue
            # Hardest positive: furthest same-class sample
            d_pos = dist[i][pos_mask[i]].max()
            # Hardest negative: closest different-class sample
            d_neg = dist[i][neg_mask[i]].min()
            loss_i = F.relu(d_pos - d_neg + self.margin)
            losses.append(loss_i)

        if not losses:
            return embeddings.sum() * 0.0
        return torch.stack(losses).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_encoder(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.07,
    verbose: bool = True,
) -> FlowEncoder:
    """
    Train the FlowEncoder with contrastive loss.

    Args:
        X: feature matrix [n_flows, n_features]
        y: integer class labels [n_flows]
        epochs: training epochs (40 is enough for a demo)
        batch_size: mini-batch size
        lr: learning rate
        temperature: NT-Xent temperature (lower = harder negatives)
        verbose: print progress

    Returns:
        Trained FlowEncoder in eval() mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    encoder = FlowEncoder(input_dim=X.shape[1]).to(device)
    criterion = ContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if verbose:
        print(f"[Encoder] Training on {len(X):,} flows  "
              f"device={device}  epochs={epochs}  batch={batch_size}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            emb = encoder(xb)
            loss = criterion(emb, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    # Restore best checkpoint
    if best_state is not None:
        encoder.load_state_dict(best_state)

    encoder.eval()
    if verbose:
        print(f"[Encoder] Training complete. Best loss: {best_loss:.4f}")

    return encoder.cpu()


# ══════════════════════════════════════════════════════════════════════════════
#  kNN CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class FlowClassifier:
    """
    k-Nearest-Neighbours classifier trained on top of frozen encoder embeddings.

    Chosen because:
    - Non-parametric: no retraining needed when new classes appear
    - Directly exploits the embedding geometry (cosine distance)
    - Interpretable: prediction = majority vote of k nearest flows
    """

    def __init__(self, k: int = 7) -> None:
        self.k = k
        self.knn = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self.class_names: list[str] = []
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "FlowClassifier":
        """Train kNN on embedding vectors with string or int labels."""
        encoded = self.label_encoder.fit_transform(labels)
        self.class_names = list(self.label_encoder.classes_)
        self.knn.fit(embeddings, encoded)
        self._fitted = True
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Return string class labels for each embedding."""
        if not self._fitted:
            raise RuntimeError("FlowClassifier must be fit before predict().")
        encoded = self.knn.predict(embeddings)
        return self.label_encoder.inverse_transform(encoded)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Return class probabilities [n_flows, n_classes]."""
        if not self._fitted:
            raise RuntimeError("FlowClassifier must be fit before predict_proba().")
        return self.knn.predict_proba(embeddings)

    def evaluate(
        self, embeddings: np.ndarray, true_labels: np.ndarray, verbose: bool = True
    ) -> dict:
        """Run evaluation and return metrics dict."""
        pred_labels = self.predict(embeddings)
        acc = accuracy_score(true_labels, pred_labels)
        report = classification_report(
            true_labels, pred_labels,
            labels=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        if verbose:
            print(f"\n[Classifier] Accuracy: {acc*100:.2f}%")
            print(classification_report(
                true_labels, pred_labels,
                labels=self.class_names,
                zero_division=0,
            ))
        return {"accuracy": acc, "report": report, "predictions": pred_labels}


# ══════════════════════════════════════════════════════════════════════════════
#  COSINE SIMILARITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingMetrics:
    """
    Computes intra-class and inter-class cosine similarity statistics.

    Target KPIs from problem statement:
      Intra-class (same type, e.g. YouTube vs Netflix): cosine sim > 0.70
      Inter-class (different types, e.g. streaming vs gaming): cosine sim < 0.30
    """

    @staticmethod
    def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
        normed = embeddings / norms
        return normed @ normed.T

    @staticmethod
    def compute(
        embeddings: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Compute per-class and pairwise-class cosine similarity.

        Returns a dict with:
          intra_class  : {class_name: mean_cosine_sim}
          inter_class  : {(class_a, class_b): mean_cosine_sim}
          mean_intra   : float
          mean_inter   : float
          kpi_intra_ok : bool  (mean_intra > 0.70)
          kpi_inter_ok : bool  (mean_inter < 0.30)
        """
        sim_matrix = EmbeddingMetrics.cosine_similarity_matrix(embeddings)
        unique_classes = sorted(set(labels))

        intra: dict[str, float] = {}
        for cls in unique_classes:
            idx = np.where(labels == cls)[0]
            if len(idx) < 2:
                intra[str(cls)] = 1.0
                continue
            sims = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    sims.append(sim_matrix[idx[i], idx[j]])
            intra[str(cls)] = float(np.mean(sims))

        inter: dict[str, float] = {}
        for i, cls_a in enumerate(unique_classes):
            for cls_b in unique_classes[i + 1:]:
                idx_a = np.where(labels == cls_a)[0]
                idx_b = np.where(labels == cls_b)[0]
                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue
                # Sample up to 200 pairs for speed
                n = min(200, len(idx_a) * len(idx_b))
                pairs_a = np.random.choice(idx_a, n)
                pairs_b = np.random.choice(idx_b, n)
                sims = [sim_matrix[a, b] for a, b in zip(pairs_a, pairs_b)]
                key = f"{cls_a} vs {cls_b}"
                inter[key] = float(np.mean(sims))

        mean_intra = float(np.mean(list(intra.values()))) if intra else 0.0
        mean_inter = float(np.mean(list(inter.values()))) if inter else 0.0

        kpi_intra_ok = mean_intra >= 0.70
        kpi_inter_ok = mean_inter <= 0.30

        if verbose:
            print("\n[Embeddings] Cosine Similarity Analysis")
            print("  ── Intra-class (same type, target > 0.70) ──")
            for cls, sim in sorted(intra.items(), key=lambda x: -x[1]):
                status = "✓" if sim >= 0.70 else "✗"
                print(f"    {status}  {cls:<25} {sim:.4f}")
            print(f"    Mean intra-class: {mean_intra:.4f}  "
                  f"{'✓ PASS' if kpi_intra_ok else '✗ FAIL'} (target ≥ 0.70)")

            print("\n  ── Inter-class (different types, target < 0.30) ──")
            for pair, sim in sorted(inter.items(), key=lambda x: x[1]):
                status = "✓" if sim <= 0.30 else "✗"
                print(f"    {status}  {pair:<40} {sim:.4f}")
            print(f"    Mean inter-class: {mean_inter:.4f}  "
                  f"{'✓ PASS' if kpi_inter_ok else '✗ FAIL'} (target ≤ 0.30)")

        return {
            "intra_class":   intra,
            "inter_class":   inter,
            "mean_intra":    mean_intra,
            "mean_inter":    mean_inter,
            "kpi_intra_ok":  kpi_intra_ok,
            "kpi_inter_ok":  kpi_inter_ok,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (shared between main.py and standalone use)
# ══════════════════════════════════════════════════════════════════════════════

ENCODER_FEATURE_COLS = [
    "duration_sec", "packets", "bytes",
    "tcp_syn_ratio", "tcp_ack_ratio", "tcp_fin_ratio",
    "inter_arrival_mean_ms", "inter_arrival_std_ms",
    "failed_connections", "dst_port",
]

PROTOCOL_MAP = {"TCP": 0, "UDP": 1, "ICMP": 2, "OTHER": 3}
DIRECTION_MAP = {"outbound": 0, "inbound": 1}


def extract_features(df) -> tuple[np.ndarray, list[str]]:
    """
    Extract and normalise features from a prepared flow DataFrame.
    Returns (X, feature_names) where X is float32 [n, features].
    """
    import pandas as pd
    feat = pd.DataFrame(index=df.index)

    for col in ENCODER_FEATURE_COLS:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            feat[col] = 0.0

    # Protocol as numeric
    if "protocol" in df.columns:
        feat["protocol_num"] = df["protocol"].map(PROTOCOL_MAP).fillna(3).astype(float)
    else:
        feat["protocol_num"] = 0.0

    # Direction as numeric
    if "direction" in df.columns:
        feat["direction_num"] = df["direction"].map(DIRECTION_MAP).fillna(0).astype(float)
    else:
        feat["direction_num"] = 0.0

    feature_names = list(feat.columns)
    X = feat.values.astype(np.float32)

    # Robust scaling (per-column median/MAD)
    medians = np.median(X, axis=0)
    mads = np.median(np.abs(X - medians), axis=0)
    mads = np.where(mads < 1e-6, 1.0, mads)
    X = (X - medians) / mads
    X = np.clip(X, -10, 10)

    return X, feature_names


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FlowEncoder — standalone training & evaluation")
    parser.add_argument("--train", action="store_true", help="Run training + evaluation demo")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.train:
        print("Running synthetic training demo...")
        np.random.seed(42)
        n_per_class = 300
        classes = ["streaming", "gaming", "voip", "browsing", "botnet"]
        Xs, ys = [], []
        for i, cls in enumerate(classes):
            center = np.random.randn(12) * 2
            X_cls = center + np.random.randn(n_per_class, 12) * 0.5
            Xs.append(X_cls.astype(np.float32))
            ys.extend([cls] * n_per_class)

        X_all = np.vstack(Xs)
        y_all = np.array(ys)

        le = LabelEncoder()
        y_int = le.fit_transform(y_all)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )
        _, _, yi_train, _ = train_test_split(
            X_all, y_int, test_size=0.2, stratify=y_int, random_state=42
        )

        encoder = train_encoder(X_train, yi_train, epochs=args.epochs,
                                batch_size=args.batch_size)

        with torch.no_grad():
            emb_train = encoder(torch.tensor(X_train)).numpy()
            emb_test  = encoder(torch.tensor(X_test)).numpy()

        clf = FlowClassifier(k=7)
        clf.fit(emb_train, y_train)
        metrics = clf.evaluate(emb_test, y_test)
        print(f"\nTest accuracy: {metrics['accuracy']*100:.2f}%")

        EmbeddingMetrics.compute(emb_test, y_test)
    else:
        print("Use --train to run the demo. Import this module from main.py for full pipeline.")