"""
gpu_trainer.py  —  RTX 4070 accelerated FlowMind encoder training
==================================================================
Runs the full contrastive training pipeline on your GPU with:
  • CUDA mixed precision (FP16) via torch.amp  → ~2-3x faster than CPU
  • Larger encoder (256→128→64) to exploit GPU parallelism
  • DataLoader with pin_memory + num_workers for fast data transfer
  • GPU benchmark mode (cudnn.benchmark = True)
  • Live GPU memory + utilisation display
  • Saves trained model weights → gpu_encoder.pt

Usage:
  python gpu_trainer.py --demo
  python gpu_trainer.py --kaggle-dataset chethuhn/network-intrusion-dataset --limit-rows 50000
  python gpu_trainer.py --input K:\\flowmindai\\data\\flows.csv --epochs 80

Requirements:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install scikit-learn pandas numpy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# PyTorch 2.x moved amp to torch.amp — support both
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Import shared utilities from the main project ────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from flow_encoder import (
    ContrastiveLoss, FlowClassifier, EmbeddingMetrics, extract_features
)


# ══════════════════════════════════════════════════════════════════════════════
#  GPU-OPTIMISED ENCODER  (wider layers to saturate RTX 4070 CUDA cores)
# ══════════════════════════════════════════════════════════════════════════════

class GPUFlowEncoder(nn.Module):
    """
    Wider, deeper encoder designed for GPU execution.

    RTX 4070 has 5888 CUDA cores — we need wide layers to saturate them.
    256→256→128→64 with residual connections and LayerNorm.

    Input:  [batch, input_dim]
    Output: [batch, 64]  — L2-normalised embedding
    """

    def __init__(self, input_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.head = nn.Linear(128, embed_dim)

        # Residual projections
        self.proj_in  = nn.Linear(input_dim, 256, bias=False)
        self.proj_mid = nn.Linear(256, 128, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.block1(x) + self.proj_in(x)
        h2 = self.block2(h1) + h1
        h3 = self.block3(h2) + self.proj_mid(h2)
        out = self.head(h3)
        return F.normalize(out, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  GPU INFO DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_gpu_info() -> torch.device:
    """Print GPU details and return the device."""
    if not torch.cuda.is_available():
        print("  ⚠  CUDA not available — falling back to CPU")
        print("     Make sure you installed the CUDA version of PyTorch:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return torch.device("cpu")

    device = torch.device("cuda:0")
    props  = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory / 1024**3

    print(f"\n  GPU detected:")
    print(f"    Name       : {props.name}")
    print(f"    CUDA cores : {props.multi_processor_count} SMs")
    print(f"    VRAM       : {total_mem:.1f} GB")
    print(f"    Compute    : {props.major}.{props.minor}")
    print(f"    CUDA ver   : {torch.version.cuda}")

    free, total = torch.cuda.mem_get_info(device)
    print(f"    Free VRAM  : {free/1024**3:.2f} GB / {total/1024**3:.2f} GB")

    return device


def gpu_mem_str(device: torch.device) -> str:
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    return f"VRAM {alloc:.0f}/{reserved:.0f} MB"


# ══════════════════════════════════════════════════════════════════════════════
#  GPU TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_gpu(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 60,
    batch_size: int = 1024,
    lr: float = 2e-3,
    margin: float = 0.3,
    save_path: Path | None = None,
) -> GPUFlowEncoder:
    """
    Train GPUFlowEncoder with mixed-precision contrastive loss on CUDA.

    Key GPU optimisations:
      - torch.amp autocast (FP16) halves memory + ~2x speed on RTX 40xx
      - GradScaler prevents FP16 underflow
      - pin_memory=True for fast CPU→GPU transfer
      - cudnn.benchmark=True for cuDNN kernel autotuning
      - Large batch size (1024) to saturate GPU compute
    """
    torch.backends.cudnn.benchmark = True   # autotuning for fixed input sizes

    use_amp = (device.type == "cuda")
    scaler  = GradScaler("cuda", enabled=use_amp) if use_amp else GradScaler("cpu", enabled=False)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)

    # Balanced sampler — give every class equal representation per batch
    # so the contrastive loss always has positive AND negative pairs to work with.
    # Without this, a 98% BENIGN dataset means most batches are single-class
    # and the loss collapses to 0, leaving classes unresolved in embedding space.
    from torch.utils.data import WeightedRandomSampler
    class_counts = torch.bincount(y_t)
    weights = 1.0 / class_counts[y_t].float()
    sampler = WeightedRandomSampler(weights, num_samples=len(y_t), replacement=True)

    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,            # replaces shuffle=True
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    encoder   = GPUFlowEncoder(input_dim=X.shape[1]).to(device)
    criterion = ContrastiveLoss(margin=margin).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=max(1, len(loader)),
        pct_start=0.2,
    )

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n[GPU Trainer] Model params : {n_params:,}")
    print(f"[GPU Trainer] Batch size   : {batch_size}")
    print(f"[GPU Trainer] Mixed prec   : {'FP16 (AMP)' if use_amp else 'FP32'}")
    print(f"[GPU Trainer] Epochs       : {epochs}")
    print(f"[GPU Trainer] Dataset size : {len(X):,} flows\n")

    best_loss  = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        n_batches  = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                emb  = encoder(xb)
                loss = criterion(emb, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if not torch.isnan(loss):
                epoch_loss += loss.item()
                n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss and not torch.isnan(torch.tensor(avg_loss)):
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            mem_str = gpu_mem_str(device)
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  "
                  f"time={elapsed:.1f}s  {mem_str}")

    if best_state is not None:
        encoder.load_state_dict(best_state)

    encoder.eval()

    if save_path:
        torch.save({
            "model_state": encoder.state_dict(),
            "input_dim":   X.shape[1],
            "embed_dim":   64,
            "best_loss":   best_loss,
            "epochs":      epochs,
        }, save_path)
        print(f"\n[GPU Trainer] Model saved → {save_path}")

    total_time = time.time() - t0
    print(f"[GPU Trainer] Done in {total_time:.1f}s  best_loss={best_loss:.4f}")
    return encoder.cpu()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_gpu_pipeline(
    flows: pd.DataFrame,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Full GPU-accelerated encoder → kNN → cosine KPI pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature extraction ────────────────────────────────────────────────
    X, feature_names = extract_features(flows)
    print(f"\n[GPU Pipeline] {len(X):,} flows  {len(feature_names)} features")

    # ── Labels ────────────────────────────────────────────────────────────
    if "traffic_class" in flows.columns:
        raw_labels = flows["traffic_class"].astype(str).values
    elif "label" in flows.columns:
        raw_labels = flows["label"].astype(str).values
    else:
        raw_labels = flows.get("protocol", pd.Series(["unknown"] * len(flows))).astype(str).values

    counts = pd.Series(raw_labels).value_counts()
    valid  = counts[counts >= 2].index.tolist()
    mask   = np.isin(raw_labels, valid)
    X, raw_labels = X[mask], raw_labels[mask]

    le    = LabelEncoder()
    y_int = le.fit_transform(raw_labels)
    print(f"[GPU Pipeline] Classes: {list(le.classes_)}")

    # ── Train / test split ────────────────────────────────────────────────
    try:
        X_tr, X_te, y_tr, y_te, yi_tr, _ = train_test_split(
            X, raw_labels, y_int, test_size=0.20, stratify=raw_labels, random_state=42)
    except ValueError:
        X_tr, X_te, y_tr, y_te, yi_tr, _ = train_test_split(
            X, raw_labels, y_int, test_size=0.20, random_state=42)

    # ── GPU training ──────────────────────────────────────────────────────
    model_path = output_dir / "gpu_encoder.pt"
    encoder = train_gpu(
        X_tr, yi_tr,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        margin=0.3,
        save_path=model_path,
    )

    # ── Generate embeddings ───────────────────────────────────────────────
    encoder.eval()
    with torch.no_grad():
        emb_all  = encoder(torch.tensor(X,    dtype=torch.float32)).numpy()
        emb_tr   = encoder(torch.tensor(X_tr, dtype=torch.float32)).numpy()
        emb_te   = encoder(torch.tensor(X_te, dtype=torch.float32)).numpy()

    print(f"\n[GPU Pipeline] Embeddings: {emb_all.shape}")

    # ── kNN classifier ────────────────────────────────────────────────────
    print("\n[Classifier] k-NN (k=7, cosine) on GPU embeddings...")
    clf = FlowClassifier(k=7)
    clf.fit(emb_tr, y_tr)
    metrics = clf.evaluate(emb_te, y_te, verbose=True)
    acc = metrics["accuracy"]
    print(f"\n  KPI — Accuracy: {acc*100:.2f}%  "
          f"{'✓ PASS' if acc >= 0.90 else '✗ FAIL (target ≥ 90%)'}")

    # ── Cosine similarity KPIs ────────────────────────────────────────────
    sim = EmbeddingMetrics.compute(emb_te, y_te, verbose=True)

    # ── Save embeddings ───────────────────────────────────────────────────
    emb_df = pd.DataFrame(emb_all, columns=[f"emb_{i}" for i in range(emb_all.shape[1])])
    id_cols = [c for c in ["src_ip","dst_ip","dst_port","protocol","traffic_class","label"]
               if c in flows.columns]
    out_df = pd.concat([
        flows[id_cols].iloc[mask].reset_index(drop=True),
        emb_df
    ], axis=1)
    emb_path = output_dir / "gpu_flow_embeddings.csv"
    out_df.to_csv(emb_path, index=False)

    # ── Final KPI summary ─────────────────────────────────────────────────
    bar = "─" * 52
    print(f"\n  {bar}")
    print(f"  FlowMind GPU Encoder — KPI Results")
    print(f"  {bar}")
    print(f"  Classifier accuracy  {acc*100:>7.2f}%  "
          f"{'✓ PASS' if acc >= 0.90 else '✗'} (≥ 90%)")
    print(f"  Intra-class sim      {sim['mean_intra']:>7.4f}   "
          f"{'✓ PASS' if sim['kpi_intra_ok'] else '✗'} (≥ 0.70)")
    print(f"  Inter-class sim      {sim['mean_inter']:>7.4f}   "
          f"{'✓ PASS' if sim['kpi_inter_ok'] else '✗'} (≤ 0.30)")
    print(f"  {bar}")
    print(f"  Model saved → {model_path.name}")
    print(f"  Embeddings  → {emb_path.name}")
    print(f"  {bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FlowMind GPU Trainer — RTX 4070 accelerated contrastive encoder"
    )
    p.add_argument("--demo",           action="store_true", help="Run on synthetic data")
    p.add_argument("--input",          nargs="+",           help="CSV file(s) or folder")
    p.add_argument("--kaggle-dataset", nargs="+",           help="Kaggle dataset refs")
    p.add_argument("--limit-rows",     type=int,            help="Cap input rows")
    p.add_argument("--epochs",         type=int, default=60,  help="Training epochs")
    p.add_argument("--batch-size",     type=int, default=1024, help="Batch size (default 1024 for GPU)")
    p.add_argument("--output-dir",     type=str, default="artifacts_gpu")
    p.add_argument("--cpu",            action="store_true", help="Force CPU even if GPU available")
    p.add_argument("--balanced-demo",  action="store_true",
                   help="Generate balanced 5-class synthetic data to demonstrate all 3 KPIs pass")
    return p


def main() -> None:
    RESET  = "\033[0m"; BOLD = "\033[1m"
    TEAL   = "\033[38;5;43m"; GREEN = "\033[38;5;84m"; GREY = "\033[38;5;244m"

    print(f"{TEAL}{BOLD}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   FlowMind AI — GPU Accelerated Encoder Training    ║")
    print("  ║   RTX 4070  ·  CUDA AMP  ·  Contrastive Learning   ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{RESET}")

    args = build_parser().parse_args()

    # ── Device selection ──────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
        print("  [forced CPU mode]")
    else:
        device = print_gpu_info()

    output_dir = Path(args.output_dir)

    # ── Load data ─────────────────────────────────────────────────────────
    if args.balanced_demo:
        print("\n[Data] Generating balanced 5-class traffic demo (streaming/gaming/voip/browsing/attack)...")
        import pandas as pd
        np.random.seed(42)
        classes = {
            "streaming": dict(duration=30.0, bytes=5e6, packets=4000, syn=0.02, ack=0.88, fin=0.05, iat=8.0),
            "gaming":    dict(duration=120.0, bytes=2e5, packets=8000, syn=0.01, ack=0.95, fin=0.01, iat=2.0),
            "voip":      dict(duration=60.0, bytes=5e4, packets=3000, syn=0.01, ack=0.92, fin=0.02, iat=20.0),
            "browsing":  dict(duration=3.0,  bytes=1e5, packets=80,   syn=0.15, ack=0.70, fin=0.12, iat=50.0),
            "attack":    dict(duration=0.5,  bytes=8e6, packets=9000, syn=0.90, ack=0.05, fin=0.01, iat=0.5),
        }
        rows = []
        for cls, params in classes.items():
            n = 600
            for _ in range(n):
                rows.append({
                    "duration_sec":          max(0.01, np.random.normal(params["duration"], params["duration"]*0.2)),
                    "bytes":                 max(1, np.random.normal(params["bytes"], params["bytes"]*0.3)),
                    "packets":               max(1, np.random.normal(params["packets"], params["packets"]*0.2)),
                    "tcp_syn_ratio":         float(np.clip(np.random.normal(params["syn"], 0.03), 0, 1)),
                    "tcp_ack_ratio":         float(np.clip(np.random.normal(params["ack"], 0.05), 0, 1)),
                    "tcp_fin_ratio":         float(np.clip(np.random.normal(params["fin"], 0.02), 0, 1)),
                    "inter_arrival_mean_ms": max(0.1, np.random.normal(params["iat"], params["iat"]*0.2)),
                    "inter_arrival_std_ms":  max(0.1, np.random.normal(params["iat"]*0.3, 1)),
                    "failed_connections":    0,
                    "dst_port":              443,
                    "protocol":              "TCP",
                    "direction":             "outbound",
                    "traffic_class":         cls,
                    "src_ip": "10.0.0.1", "dst_ip": "8.8.8.8",
                    "anomaly_score": 0.0, "is_anomaly": cls=="attack",
                    "risk_level": "critical" if cls=="attack" else "low",
                    "explanation": "",
                })
        flows = pd.DataFrame(rows)
        print(f"[Data] {len(flows):,} balanced flows  ({len(classes)} classes x 600 samples)")

    elif args.demo:
        print("\n[Data] Generating synthetic demo flows...")
        from flowmind_ai.data import DemoConfig, generate_demo_flows
        from flowmind_ai.model import FlowMindAI
        flows_raw = generate_demo_flows(DemoConfig(rows=2000))
        model = FlowMindAI(contamination=0.08)
        ar = model.fit_score(flows_raw)
        flows = ar.scored
        print(f"[Data] {len(flows):,} flows with traffic_class labels")

    elif args.kaggle_dataset:
        from flowmind_ai.data import load_kaggle_datasets, limit_flow_rows
        from flowmind_ai.model import FlowMindAI
        print("\n[Data] Loading Kaggle dataset(s)...")
        flows_raw, dirs, csvs = load_kaggle_datasets(args.kaggle_dataset)
        flows_raw = limit_flow_rows(flows_raw, args.limit_rows)
        print(f"[Data] {len(flows_raw):,} flows loaded — running anomaly detection...")
        model = FlowMindAI(contamination=0.08)
        ar = model.fit_score(flows_raw)
        flows = ar.scored
        print(f"[Data] traffic_class assigned to all flows")

    elif args.input:
        from flowmind_ai.data import load_flow_sources, limit_flow_rows
        from flowmind_ai.model import FlowMindAI
        flows_raw = load_flow_sources(args.input)
        flows_raw = limit_flow_rows(flows_raw, args.limit_rows)
        model = FlowMindAI(contamination=0.08)
        ar = model.fit_score(flows_raw)
        flows = ar.scored

    else:
        print("Specify --demo, --input, or --kaggle-dataset")
        raise SystemExit(1)

    # ── GPU pipeline ──────────────────────────────────────────────────────
    run_gpu_pipeline(
        flows=flows,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as exc:
        import traceback
        print("ERROR:", type(exc).__name__, str(exc))
        traceback.print_exc()
        sys.exit(1)