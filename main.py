from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlowMind AI - Context-Aware Flow Embeddings for Adaptive Network Traffic Classification and Threat Detection"
    )
    parser.add_argument("--generate-dataset", action="store_true")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--kaggle-dataset", nargs="+")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--contamination", type=float, default=0.08)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--auto-learn", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--memory-size", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--per-source-model", action="store_true")
    parser.add_argument("--encoder-epochs", type=int, default=40,
                        help="Contrastive training epochs for the neural encoder (default: 40)")
    parser.add_argument("--skip-encoder", action="store_true",
                        help="Skip neural encoder training (faster, anomaly detection only)")
    return parser


def run_per_source_models(flows, contamination, auto_learn, batch_size,
                          warmup_batches, memory_size, progress_every):
    from flowmind_ai.model import AdaptiveArtifacts, FlowMindAI, ScoreArtifacts

    grouped = list(flows.groupby("dataset_source", sort=True))
    scored_parts, batch_parts, source_rows = [], [], []

    for source_name, source_frame in grouped:
        print(f"[FlowMind] Training source-specific model for: {source_name} ({len(source_frame)} flows)")
        model = FlowMindAI(contamination=contamination)
        if auto_learn:
            source_result = model.adaptive_fit_score(
                source_frame,
                batch_size=batch_size,
                warmup_batches=warmup_batches,
                memory_size=memory_size,
                progress_every=progress_every,
            )
            batch_summary = source_result.batch_summaries.copy()
            batch_summary["dataset_source"] = source_name
            batch_parts.append(batch_summary)
        else:
            source_result = model.fit_score(source_frame)

        scored_frame = source_result.scored.copy()
        scored_frame["dataset_source"] = source_name
        scored_parts.append(scored_frame)

        src_sum = dict(source_result.summary)
        src_sum["dataset_source"] = source_name
        src_sum["threshold"] = round(float(source_result.threshold), 4)
        source_rows.append(src_sum)

    scored = (
        flows.iloc[0:0].copy()
        if not scored_parts
        else pd.concat(scored_parts, ignore_index=True)
    )
    scored = scored.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    summary = {
        "total_flows":   int(len(scored)),
        "flagged_flows": int(scored["is_anomaly"].sum()),
        "flag_rate":     round(float(scored["is_anomaly"].mean() * 100), 2),
        "mean_score":    round(float(scored["anomaly_score"].mean()), 4),
        "max_score":     round(float(scored["anomaly_score"].max()), 4),
        "top_protocol":  str(scored["protocol"].mode().iloc[0]) if not scored.empty else "N/A",
        "source_models": int(len(grouped)),
    }

    if "traffic_class" in scored.columns:
        summary["class_breakdown"] = scored["traffic_class"].value_counts().to_dict()

    if "label" in scored.columns:
        known_attack_mask = ~scored["label"].astype(str).isin(["normal", "benign", "0", "unknown"])
        known_normal_mask =  scored["label"].astype(str).isin(["normal", "benign", "0"])
        summary["known_attack_flows"] = int(known_attack_mask.sum())
        summary["known_normal_flows"] = int(known_normal_mask.sum())
        if int(known_attack_mask.sum()) > 0:
            detected = int((scored["is_anomaly"] & known_attack_mask).sum())
            summary["detected_known_attacks"] = detected
            summary["known_attack_recall"] = round(detected / int(known_attack_mask.sum()) * 100, 2)
        if int(known_normal_mask.sum()) > 0:
            fp = int((scored["is_anomaly"] & known_normal_mask).sum())
            summary["benign_false_positives"] = fp
            summary["benign_false_positive_rate"] = round(fp / int(known_normal_mask.sum()) * 100, 2)

    summary["per_source_thresholds"] = ", ".join(
        f"{r['dataset_source']}={r['threshold']}" for r in source_rows
    )

    if auto_learn:
        return AdaptiveArtifacts(
            scored=scored,
            threshold=float(scored["anomaly_score"].quantile(1 - contamination)) if not scored.empty else 0.0,
            summary=summary,
            batch_summaries=pd.concat(batch_parts, ignore_index=True) if batch_parts else pd.DataFrame(),
        )
    return ScoreArtifacts(
        scored=scored,
        threshold=float(scored["anomaly_score"].quantile(1 - contamination)) if not scored.empty else 0.0,
        summary=summary,
    )


def run_encoder_pipeline(
    flows: pd.DataFrame,
    output_dir: Path,
    encoder_epochs: int,
) -> None:
    """
    Full neural encoder pipeline:
      1. Extract features from flow DataFrame
      2. Train FlowEncoder with contrastive loss
      3. Generate embeddings for all flows
      4. Train kNN classifier on embeddings
      5. Measure cosine similarity (intra vs inter class)
      6. Export embeddings CSV + print KPI report
    """
    try:
        import torch
        from flow_encoder import (
            FlowEncoder, train_encoder,
            FlowClassifier, EmbeddingMetrics,
            extract_features,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        print(f"[Encoder] Skipping neural encoder — missing dependency: {exc}")
        print("  Install with: pip install torch scikit-learn")
        return

    print("\n[Encoder] ── Neural Encoder Pipeline ──────────────────────────")

    # ── 1. Feature extraction ─────────────────────────────────────────────
    X, feature_names = extract_features(flows)
    print(f"[Encoder] Features: {len(feature_names)} dims  Flows: {len(X):,}")

    # ── 2. Build training labels from traffic_class or label column ───────
    if "traffic_class" in flows.columns:
        raw_labels = flows["traffic_class"].astype(str).values
    elif "label" in flows.columns:
        raw_labels = flows["label"].astype(str).values
    else:
        # Fall back to protocol as a weak label signal
        raw_labels = flows["protocol"].astype(str).values if "protocol" in flows.columns \
                     else np.array(["unknown"] * len(flows))

    le = LabelEncoder()
    y_int = le.fit_transform(raw_labels)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    print(f"[Encoder] Classes ({n_classes}): {', '.join(class_names[:8])}"
          f"{'…' if n_classes > 8 else ''}")

    # Need at least 2 samples per class for contrastive loss
    class_counts = pd.Series(raw_labels).value_counts()
    valid_classes = class_counts[class_counts >= 2].index.tolist()
    if len(valid_classes) < 2:
        print("[Encoder] Not enough class diversity for contrastive training — skipping.")
        return

    mask = np.isin(raw_labels, valid_classes)
    X_filtered = X[mask]
    y_filtered = raw_labels[mask]
    y_int_filtered = le.transform(y_filtered)

    # ── 3. Train/test split ───────────────────────────────────────────────
    # Use stratify only if every class has ≥ 2 samples
    try:
        X_train, X_test, y_train, y_test, yi_train, _ = train_test_split(
            X_filtered, y_filtered, y_int_filtered,
            test_size=0.20, stratify=y_filtered, random_state=42,
        )
    except ValueError:
        X_train, X_test, y_train, y_test, yi_train, _ = train_test_split(
            X_filtered, y_filtered, y_int_filtered,
            test_size=0.20, random_state=42,
        )

    # ── 4. Train encoder with contrastive loss ────────────────────────────
    encoder = train_encoder(
        X_train, yi_train,
        epochs=encoder_epochs,
        batch_size=min(256, len(X_train)),
        verbose=True,
    )

    # ── 5. Generate embeddings for ALL flows ──────────────────────────────
    encoder.eval()
    with torch.no_grad():
        all_embeddings = encoder(torch.tensor(X, dtype=torch.float32)).numpy()
        train_embeddings = encoder(torch.tensor(X_train, dtype=torch.float32)).numpy()
        test_embeddings  = encoder(torch.tensor(X_test,  dtype=torch.float32)).numpy()

    # ── 6. kNN classifier ─────────────────────────────────────────────────
    print("\n[Classifier] Training k-NN (k=7, cosine distance) on embeddings...")
    clf = FlowClassifier(k=7)
    clf.fit(train_embeddings, y_train)
    metrics = clf.evaluate(test_embeddings, y_test, verbose=True)
    acc = metrics["accuracy"]

    kpi_acc_ok = acc >= 0.90
    print(f"\n  KPI — Classification accuracy: {acc*100:.2f}%  "
          f"{'✓ PASS' if kpi_acc_ok else '✗ FAIL (target ≥ 90%)'}")

    # ── 7. Cosine similarity KPI ──────────────────────────────────────────
    sim_metrics = EmbeddingMetrics.compute(test_embeddings, y_test, verbose=True)

    # ── 8. Export embeddings CSV ──────────────────────────────────────────
    emb_df = pd.DataFrame(
        all_embeddings,
        columns=[f"emb_{i}" for i in range(all_embeddings.shape[1])],
    )
    id_cols = [c for c in ["src_ip", "dst_ip", "dst_port", "protocol",
                            "traffic_class", "label"] if c in flows.columns]
    emb_df = pd.concat([flows[id_cols].reset_index(drop=True), emb_df], axis=1)
    emb_path = output_dir / "flow_embeddings.csv"
    emb_df.to_csv(emb_path, index=False)
    print(f"\n[Encoder] Embeddings saved: {emb_path}  shape={all_embeddings.shape}")

    # ── 9. Summary ────────────────────────────────────────────────────────
    print("\n[Encoder] ── KPI Summary ───────────────────────────────────────")
    print(f"  Classifier accuracy  : {acc*100:.2f}%  "
          f"{'✓' if kpi_acc_ok else '✗'} (target ≥ 90%)")
    print(f"  Mean intra-class sim : {sim_metrics['mean_intra']:.4f}  "
          f"{'✓' if sim_metrics['kpi_intra_ok'] else '✗'} (target ≥ 0.70)")
    print(f"  Mean inter-class sim : {sim_metrics['mean_inter']:.4f}  "
          f"{'✓' if sim_metrics['kpi_inter_ok'] else '✗'} (target ≤ 0.30)")
    print("[Encoder] ─────────────────────────────────────────────────────\n")


def main() -> None:
    try:
        from flowmind_ai.data import (
            DemoConfig, generate_demo_dataset, generate_demo_flows,
            limit_flow_rows, load_kaggle_datasets, load_flow_sources,
        )
        from flowmind_ai.model import FlowMindAI
        from flowmind_ai.reporting import write_html_report
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        print(
            "Missing dependency detected.\n"
            f"Python could not import: {missing}\n\n"
            "Install the project dependencies with:\n"
            "  python -m pip install -r requirements.txt\n\n"
            f"Active interpreter: {sys.executable}"
        )
        raise SystemExit(1) from exc

    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_dataset:
        from flowmind_ai.data import generate_demo_dataset
        created_files = generate_demo_dataset(
            output_dir=output_dir,
            days=max(1, args.days),
            rows_per_day=max(50, args.rows),
            anomaly_fraction=args.contamination,
        )
        print(f"Generated {len(created_files)} demo CSV files in: {output_dir}")
        for fp in created_files:
            print(f"- {fp}")
        return

    # ── Load data ─────────────────────────────────────────────────────────
    if args.kaggle_dataset:
        try:
            flows, dataset_dirs, csv_files = load_kaggle_datasets(args.kaggle_dataset)
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            print(f"Kaggle dataset error: {exc}")
            raise SystemExit(1) from exc
        print("Downloaded Kaggle dataset(s) to:")
        for d in dataset_dirs:
            print(f"- {d}")
        print(f"Loaded {len(flows)} flows from {len(csv_files)} CSV file(s)")
    elif args.demo or not args.input:
        flows = generate_demo_flows(DemoConfig(rows=args.rows))
        csv_path = output_dir / "demo_flows.csv"
        flows.to_csv(csv_path, index=False)
        print(f"Generated demo traffic: {csv_path}")
    else:
        try:
            flows = load_flow_sources(args.input)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Input error: {exc}")
            raise SystemExit(1) from exc
        print(f"Loaded {len(flows)} flows from: {', '.join(args.input)}")

    flows = limit_flow_rows(flows, args.limit_rows)
    if args.limit_rows:
        print(f"Using {len(flows)} flow rows after applying --limit-rows")

    # ── Anomaly detection + rule-based classification ─────────────────────
    if args.per_source_model and "dataset_source" in flows.columns:
        analysis_result = run_per_source_models(
            flows=flows,
            contamination=args.contamination,
            auto_learn=args.auto_learn,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            memory_size=args.memory_size,
            progress_every=args.progress_every,
        )
        if getattr(analysis_result, "batch_summaries", None) is not None:
            batch_path = output_dir / "adaptive_learning_summary.csv"
            analysis_result.batch_summaries.to_csv(batch_path, index=False)
            print(f"Adaptive learning summary: {batch_path}")
    else:
        model = FlowMindAI(contamination=args.contamination)
        if args.auto_learn:
            analysis_result = model.adaptive_fit_score(
                flows,
                batch_size=args.batch_size,
                warmup_batches=args.warmup_batches,
                memory_size=args.memory_size,
                progress_every=args.progress_every,
            )
            batch_path = output_dir / "adaptive_learning_summary.csv"
            analysis_result.batch_summaries.to_csv(batch_path, index=False)
            print(f"Adaptive learning summary: {batch_path}")
        else:
            analysis_result = model.fit_score(flows)

    # ── Neural encoder pipeline (contrastive + kNN + cosine KPIs) ─────────
    if not args.skip_encoder:
        # Use the scored frame (has traffic_class column) as input
        run_encoder_pipeline(
            flows=analysis_result.scored,
            output_dir=output_dir,
            encoder_epochs=args.encoder_epochs,
        )

    # ── HTML report ───────────────────────────────────────────────────────
    report_path = write_html_report(
        analysis_result.scored,
        analysis_result.summary,
        output_dir / "flowmind_report.html",
    )

    # ── Terminal summary ──────────────────────────────────────────────────
    dedup_cols = [c for c in ["src_ip", "dst_ip", "dst_port", "protocol"]
                  if c in analysis_result.scored.columns]
    top_hits = (
        analysis_result.scored
        .sort_values("anomaly_score", ascending=False)
        .drop_duplicates(subset=dedup_cols if dedup_cols else None)
        .head(5)
    )
    has_class = "traffic_class" in analysis_result.scored.columns

    print("\nFlowMind AI summary")
    for key, value in analysis_result.summary.items():
        if isinstance(value, dict):
            print(f"- {key}:")
            for cls, cnt in sorted(value.items(), key=lambda x: -x[1]):
                print(f"    {cls}: {cnt}")
        else:
            print(f"- {key}: {value}")
    print(f"- threshold: {analysis_result.threshold:.4f}")
    print(f"- report: {report_path}")

    print("\nTop suspicious flows")
    for row in top_hits.itertuples():
        class_str = f" class={row.traffic_class}" if has_class else ""
        print(
            f"- {row.src_ip} -> {row.dst_ip}:{row.dst_port} "
            f"[{row.protocol}] score={row.anomaly_score:.3f} "
            f"risk={row.risk_level}{class_str} explanation={row.explanation}"
        )


if __name__ == "__main__":
    main()