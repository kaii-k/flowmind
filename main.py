from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlowMind AI - Offline behavioral threat detection MVP"
    )
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Generate multiple demo CSVs into the output directory and exit",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of demo CSV files to generate with --generate-dataset",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="One or more flow CSV files, or a folder containing CSV files",
    )
    parser.add_argument(
        "--kaggle-dataset",
        nargs="+",
        help="One or more Kaggle dataset references, for example chethuhn/network-intrusion-dataset",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic flow data and run the detector",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=600,
        help="Number of demo rows to generate when using --demo",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.08,
        help="Expected anomaly fraction used to set the decision threshold",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory for generated CSV and HTML report",
    )
    parser.add_argument(
        "--auto-learn",
        action="store_true",
        help="Score traffic in batches and continuously refit on likely-benign traffic",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for adaptive learning mode",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Number of initial batches used to bootstrap the adaptive model",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=5000,
        help="Maximum number of likely-benign flows retained for adaptive relearning",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print adaptive-learning progress every N batches",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        help="Optional cap on rows loaded from the resolved dataset for faster testing",
    )
    parser.add_argument(
        "--per-source-model",
        action="store_true",
        help="Train separate baselines per dataset_source and merge the scored output",
    )
    return parser


def run_per_source_models(
    flows,
    contamination: float,
    auto_learn: bool,
    batch_size: int,
    warmup_batches: int,
    memory_size: int,
    progress_every: int,
):
    from flowmind_ai.model import AdaptiveArtifacts, FlowMindAI, ScoreArtifacts

    grouped = list(flows.groupby("dataset_source", sort=True))
    scored_parts = []
    batch_parts = []
    source_rows = []

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

        source_summary = dict(source_result.summary)
        source_summary["dataset_source"] = source_name
        source_summary["threshold"] = round(float(source_result.threshold), 4)
        source_rows.append(source_summary)

    scored = flows.iloc[0:0].copy() if not scored_parts else pd.concat(scored_parts, ignore_index=True)
    scored = scored.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    summary = {
        "total_flows": int(len(scored)),
        "flagged_flows": int(scored["is_anomaly"].sum()),
        "flag_rate": round(float(scored["is_anomaly"].mean() * 100), 2),
        "mean_score": round(float(scored["anomaly_score"].mean()), 4),
        "max_score": round(float(scored["anomaly_score"].max()), 4),
        "top_protocol": str(scored["protocol"].mode().iloc[0]) if not scored.empty else "N/A",
        "source_models": int(len(grouped)),
    }
    if "label" in scored.columns:
        known_attack_mask = ~scored["label"].astype(str).isin(["normal", "benign", "0", "unknown"])
        known_normal_mask = scored["label"].astype(str).isin(["normal", "benign", "0"])
        summary["known_attack_flows"] = int(known_attack_mask.sum())
        summary["known_normal_flows"] = int(known_normal_mask.sum())
        if int(known_attack_mask.sum()) > 0:
            detected = int((scored["is_anomaly"] & known_attack_mask).sum())
            summary["detected_known_attacks"] = detected
            summary["known_attack_recall"] = round((detected / int(known_attack_mask.sum())) * 100, 2)
        if int(known_normal_mask.sum()) > 0:
            benign_flagged = int((scored["is_anomaly"] & known_normal_mask).sum())
            summary["benign_false_positives"] = benign_flagged
            summary["benign_false_positive_rate"] = round((benign_flagged / int(known_normal_mask.sum())) * 100, 2)

    summary["per_source_thresholds"] = ", ".join(
        f"{row['dataset_source']}={row['threshold']}"
        for row in source_rows
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


def main() -> None:
    try:
        from flowmind_ai.data import (
            DemoConfig,
            generate_demo_dataset,
            generate_demo_flows,
            limit_flow_rows,
            load_kaggle_datasets,
            load_flow_sources,
        )
        from flowmind_ai.model import FlowMindAI
        from flowmind_ai.reporting import write_html_report
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        print(
            "Missing dependency detected.\n"
            f"Python could not import: {missing}\n\n"
            "Install the project dependencies in your active interpreter with:\n"
            "  python -m pip install -r requirements.txt\n\n"
            f"Active interpreter: {sys.executable}\n\n"
            "If you are using an IDE, make sure it is pointing to the same Python "
            "environment where those packages are installed."
        )
        raise SystemExit(1) from exc

    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_dataset:
        created_files = generate_demo_dataset(
            output_dir=output_dir,
            days=max(1, args.days),
            rows_per_day=max(50, args.rows),
            anomaly_fraction=args.contamination,
        )
        print(f"Generated {len(created_files)} demo CSV files in: {output_dir}")
        for file_path in created_files:
            print(f"- {file_path}")
        return

    if args.kaggle_dataset:
        try:
            flows, dataset_dirs, csv_files = load_kaggle_datasets(args.kaggle_dataset)
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            print(
                f"Kaggle dataset error: {exc}\n\n"
                "Example:\n"
                "  python main.py --kaggle-dataset chethuhn/network-intrusion-dataset mrwellsdavid/unsw-nb15 arnobbhowmik/ton-iot-network-dataset --auto-learn --batch-size 1000"
            )
            raise SystemExit(1) from exc

        print("Downloaded Kaggle dataset(s) to:")
        for dataset_dir in dataset_dirs:
            print(f"- {dataset_dir}")
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
            print(
                f"Input error: {exc}\n\n"
                "Examples:\n"
                "  python main.py --input artifacts\\demo_flows.csv --output-dir artifacts\n"
                "  python main.py --input data\\monday.csv data\\tuesday.csv --output-dir artifacts\n"
                "  python main.py --input data\\ --output-dir artifacts"
            )
            raise SystemExit(1) from exc

        print(f"Loaded {len(flows)} flows from: {', '.join(args.input)}")

    flows = limit_flow_rows(flows, args.limit_rows)
    if args.limit_rows:
        print(f"Using {len(flows)} flow rows after applying --limit-rows")

    if args.per_source_model and "dataset_source" in flows.columns:
        result = run_per_source_models(
            flows=flows,
            contamination=args.contamination,
            auto_learn=args.auto_learn,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            memory_size=args.memory_size,
            progress_every=args.progress_every,
        )
        batch_summary_path = output_dir / "adaptive_learning_summary.csv"
        if getattr(result, "batch_summaries", None) is not None:
            result.batch_summaries.to_csv(batch_summary_path, index=False)
            print(f"Adaptive learning summary: {batch_summary_path}")
    else:
        model = FlowMindAI(contamination=args.contamination)
        if args.auto_learn:
            result = model.adaptive_fit_score(
                flows,
                batch_size=args.batch_size,
                warmup_batches=args.warmup_batches,
                memory_size=args.memory_size,
                progress_every=args.progress_every,
            )
            batch_summary_path = output_dir / "adaptive_learning_summary.csv"
            result.batch_summaries.to_csv(batch_summary_path, index=False)
            print(f"Adaptive learning summary: {batch_summary_path}")
        else:
            result = model.fit_score(flows)

    report_path = write_html_report(
        result.scored,
        result.summary,
        output_dir / "flowmind_report.html",
    )

    top_hits = result.scored.sort_values("anomaly_score", ascending=False).head(5)

    print("\nFlowMind AI summary")
    for key, value in result.summary.items():
        print(f"- {key}: {value}")
    print(f"- threshold: {result.threshold:.4f}")
    print(f"- report: {report_path}")

    print("\nTop suspicious flows")
    for row in top_hits.itertuples():
        print(
            f"- {row.src_ip} -> {row.dst_ip}:{row.dst_port} "
            f"[{row.protocol}] score={row.anomaly_score:.3f} "
            f"risk={row.risk_level} explanation={row.explanation}"
        )


if __name__ == "__main__":
    main()
