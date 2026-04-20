# FlowMind AI

FlowMind AI is an offline behavioral threat detection MVP for network flow analysis.
It learns normal traffic patterns from flow-level telemetry, scores anomalous flows,
and produces an explainable HTML report for analysts.

## What This Starter Includes

- Offline anomaly detection using only `numpy` and `pandas`
- Context-aware flow embeddings
- Per-flow explanations with top contributing features
- Synthetic traffic generator for demo/testing
- Optional Kaggle dataset download with `kagglehub`
- Adaptive learning loop for batch-by-batch retraining
- CLI entrypoint you can extend in your IDE
- Multi-dataset Kaggle ingestion for CICIDS2017, UNSW-NB15, and TON-IoT style CSVs

## Project Structure

- `main.py` - CLI entrypoint
- `flowmind_ai/data.py` - demo flow generation and CSV loading
- `flowmind_ai/model.py` - embedding, training, scoring, explanations
- `flowmind_ai/reporting.py` - HTML report writer

## Expected CSV Shape

If you want to test with your own data, provide one CSV, multiple CSVs, or a folder
containing CSV files with some or all of these columns:

- `timestamp`
- `src_ip`
- `dst_ip`
- `src_port`
- `dst_port`
- `protocol`
- `duration_sec`
- `packets`
- `bytes`
- `tcp_syn_ratio`
- `tcp_ack_ratio`
- `tcp_fin_ratio`
- `inter_arrival_mean_ms`
- `inter_arrival_std_ms`
- `failed_connections`
- `direction`

Missing numeric columns are auto-filled with `0`, and missing categoricals are filled
with sensible defaults.

## Quick Start

Install the dependencies in the Python interpreter you plan to use:

```powershell
python -m pip install -r requirements.txt
```

Use the bundled runtime if your local Python does not already have `numpy` and `pandas`.

```powershell
python main.py --demo --rows 600 --output-dir artifacts
```

This generates:

- `artifacts/demo_flows.csv`
- `artifacts/flowmind_report.html`

## Example Commands

```powershell
python main.py --generate-dataset --days 5 --rows 1000 --output-dir data
python main.py --demo
python main.py --demo --rows 1000 --contamination 0.07
python main.py --input path\to\flows.csv --output-dir artifacts
python main.py --input data\monday.csv data\tuesday.csv --output-dir artifacts
python main.py --input data\ --output-dir artifacts
python main.py --input data\ --auto-learn --batch-size 1000 --output-dir artifacts
python main.py --kaggle-dataset chethuhn/network-intrusion-dataset --auto-learn --batch-size 1000 --output-dir artifacts
python main.py --kaggle-dataset chethuhn/network-intrusion-dataset mrwellsdavid/unsw-nb15 arnobbhowmik/ton-iot-network-dataset --auto-learn --batch-size 5000 --progress-every 5 --limit-rows 100000 --output-dir artifacts
python main.py --kaggle-dataset chethuhn/network-intrusion-dataset --auto-learn --batch-size 5000 --progress-every 5 --limit-rows 100000 --output-dir artifacts
```

## How The Model Works

1. Clean and normalize flow-level traffic data
2. Build compact context-aware embeddings from rates, ratios, and network context
3. Learn a robust baseline using median and MAD scaling
4. Score anomalies using:
   - baseline distance
   - reconstruction error
   - context rarity
5. Explain top anomalous features per flow

## Adaptive Learning

When you run with `--auto-learn`, FlowMind AI:

1. Uses the first batch as a warmup baseline
2. Scores the next batch
3. Keeps likely-benign flows from that batch
4. Re-fits the model on a rolling memory buffer
5. Repeats automatically for the rest of the dataset

This gives you a simple online-learning style loop for evolving traffic patterns.

## Good Next Steps

- Replace demo data with CICIDS2017 or UNSW-NB15 flow CSV exports
- Add packet-to-flow conversion using `scapy` or `pyshark`
- Swap the custom detector for `IsolationForest` when `scikit-learn` is available
- Add a Streamlit dashboard once that dependency is installed
