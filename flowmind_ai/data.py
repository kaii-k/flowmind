from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
from typing import Iterable
import random
import re

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "duration_sec",
    "packets",
    "bytes",
    "tcp_syn_ratio",
    "tcp_ack_ratio",
    "tcp_fin_ratio",
    "inter_arrival_mean_ms",
    "inter_arrival_std_ms",
    "failed_connections",
]

CATEGORICAL_COLUMNS = [
    "protocol",
    "direction",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
]

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date", "ts"],
    "src_ip": ["src_ip", "sourceip", "source_ip", "srcip", "source", "src host"],
    "dst_ip": ["dst_ip", "destinationip", "destination_ip", "dstip", "destination", "dst host"],
    "src_port": ["src_port", "sourceport", "source_port", "sport", "srcport", "src_port_num"],
    "dst_port": ["dst_port", "destinationport", "destination_port", "dport", "dstport", "dest_port"],
    "protocol": ["protocol", "proto", "protocol_type"],
    "direction": ["direction", "flow_direction"],
    "state": ["state", "conn_state"],
    "duration_sec": ["duration_sec", "duration", "flow_duration", "dur", "duration_seconds"],
    "packets": [
        "packets",
        "packet_count",
        "total_packets",
        "pkts",
        "tot_pkts",
        "flow_packets",
        "total_fwd_packets",
        "total_backward_packets",
        "subflow_fwd_packets",
        "subflow_bwd_packets",
        "spkts",
        "dpkts",
        "src_pkts",
        "dst_pkts",
    ],
    "bytes": [
        "bytes",
        "byte_count",
        "total_bytes",
        "flow_bytes",
        "tot_bytes",
        "payload_bytes",
        "total_length_of_fwd_packets",
        "total_length_of_bwd_packets",
        "subflow_fwd_bytes",
        "subflow_bwd_bytes",
        "sbytes",
        "dbytes",
        "src_bytes",
        "dst_bytes",
        "src_ip_bytes",
        "dst_ip_bytes",
    ],
    "tcp_syn_ratio": ["tcp_syn_ratio", "syn_ratio", "synrate", "syn_rate"],
    "tcp_ack_ratio": ["tcp_ack_ratio", "ack_ratio", "ackrate", "ack_rate"],
    "tcp_fin_ratio": ["tcp_fin_ratio", "fin_ratio", "finrate", "fin_rate"],
    "syn_flag_count": ["syn_flag_count", "syn_flag", "syn_flag_count"],
    "ack_flag_count": ["ack_flag_count", "ack_flag", "ack_flag_count"],
    "fin_flag_count": ["fin_flag_count", "fin_flag", "fin_flag_count"],
    "inter_arrival_mean_ms": [
        "inter_arrival_mean_ms",
        "iat_mean",
        "flow_iat_mean",
        "mean_iat",
        "interarrival_mean",
    ],
    "inter_arrival_std_ms": [
        "inter_arrival_std_ms",
        "iat_std",
        "flow_iat_std",
        "std_iat",
        "interarrival_std",
    ],
    "failed_connections": ["failed_connections", "failed_conn", "failed_attempts", "connection_errors"],
    "label": ["label", "attack", "attack_type", "class", "target", "is_attack", "attack_cat", "type"],
}

PROTOCOL_NUMBER_MAP = {
    "1": "ICMP",
    "6": "TCP",
    "17": "UDP",
}

PROTOCOL_CANONICAL_MAP = {
    "TCP": "TCP",
    "UDP": "UDP",
    "ICMP": "ICMP",
    "ICMPV6": "ICMP",
    "IGMP": "OTHER",
    "SCTP": "OTHER",
    "GRE": "OTHER",
    "ESP": "OTHER",
    "AH": "OTHER",
    "OSPF": "OTHER",
    "IP": "OTHER",
    "IPIP": "OTHER",
    "PIM": "OTHER",
    "GGP": "OTHER",
    "EGP": "OTHER",
}

SKIP_FILE_PATTERNS = (
    "feature",
    "list_event",
    "event",
    "readme",
    "description",
    "metadata",
    "schema",
)


@dataclass(frozen=True)
class DemoConfig:
    rows: int = 500
    anomaly_fraction: float = 0.08
    seed: int = 7


def _sample_internal_ip(rng: random.Random) -> str:
    return f"10.0.{rng.randint(0, 7)}.{rng.randint(2, 250)}"


def _sample_external_ip(rng: random.Random) -> str:
    return f"{rng.randint(23, 191)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _weighted_choice(rng: random.Random, options: Iterable[tuple[str, float]]) -> str:
    values, weights = zip(*options)
    return rng.choices(values, weights=weights, k=1)[0]


def generate_demo_flows(config: DemoConfig) -> pd.DataFrame:
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    protocols = [("TCP", 0.72), ("UDP", 0.22), ("ICMP", 0.06)]
    services = [80, 443, 53, 22, 3389, 8080, 123]

    rows: list[dict[str, object]] = []
    anomaly_count = max(1, int(config.rows * config.anomaly_fraction))
    anomaly_indexes = set(rng.sample(range(config.rows), anomaly_count))

    for idx in range(config.rows):
        anomalous = idx in anomaly_indexes
        protocol = _weighted_choice(rng, protocols)
        direction = "outbound" if rng.random() > 0.42 else "inbound"

        src_ip = _sample_internal_ip(rng)
        dst_ip = _sample_external_ip(rng) if direction == "outbound" else _sample_internal_ip(rng)
        src_port = rng.randint(1024, 65535)
        dst_port = rng.choice(services)

        base_duration = float(np.clip(np_rng.lognormal(mean=0.6, sigma=0.55), 0.08, 45.0))
        base_packets = int(np.clip(np_rng.normal(loc=36, scale=18), 4, 320))
        avg_pkt_size = float(np.clip(np_rng.normal(loc=620, scale=160), 80, 1500))
        base_bytes = int(base_packets * avg_pkt_size)
        syn_ratio = float(np.clip(np_rng.normal(loc=0.15, scale=0.07), 0.0, 1.0))
        ack_ratio = float(np.clip(np_rng.normal(loc=0.48, scale=0.12), 0.0, 1.0))
        fin_ratio = float(np.clip(np_rng.normal(loc=0.11, scale=0.06), 0.0, 1.0))
        iat_mean = float(np.clip(np_rng.normal(loc=22, scale=10), 1, 800))
        iat_std = float(np.clip(np_rng.normal(loc=14, scale=8), 0.5, 600))
        failed_connections = 0 if rng.random() > 0.06 else 1

        if protocol == "UDP":
            syn_ratio = 0.0
            ack_ratio = 0.0
            fin_ratio = 0.0
            iat_mean *= 0.7
        elif protocol == "ICMP":
            base_packets = int(np.clip(np_rng.normal(loc=12, scale=6), 1, 90))
            avg_pkt_size = float(np.clip(np_rng.normal(loc=180, scale=60), 64, 512))
            base_bytes = int(base_packets * avg_pkt_size)
            syn_ratio = ack_ratio = fin_ratio = 0.0

        if anomalous:
            anomaly_type = rng.choice(["burst_exfil", "slow_scan", "beaconing"])
            dst_port = rng.choice([21, 23, 25, 445, 4444, 5555, 31337])

            if anomaly_type == "burst_exfil":
                base_duration *= 1.9
                base_packets = int(base_packets * 5.5)
                base_bytes = int(base_bytes * 8.0)
                ack_ratio = float(np.clip(ack_ratio * 0.45, 0, 1))
                iat_mean *= 0.3
                iat_std *= 0.5
            elif anomaly_type == "slow_scan":
                base_duration *= 4.2
                base_packets = int(np.clip(base_packets * 0.45, 2, 40))
                base_bytes = int(np.clip(base_bytes * 0.25, 120, 9000))
                syn_ratio = float(np.clip(syn_ratio + 0.55, 0, 1))
                ack_ratio = float(np.clip(ack_ratio * 0.2, 0, 1))
                failed_connections = rng.randint(3, 9)
                iat_mean *= 3.5
                iat_std *= 2.8
            else:
                base_duration *= 2.8
                base_packets = int(np.clip(base_packets * 0.8, 3, 120))
                base_bytes = int(np.clip(base_bytes * 0.55, 150, 25000))
                iat_mean = float(np.clip(iat_mean * 5.0, 80, 2500))
                iat_std = float(np.clip(iat_std * 0.4, 1, 120))
                failed_connections = rng.randint(1, 4)

        rows.append(
            {
                "timestamp": pd.Timestamp("2026-04-20") + pd.Timedelta(seconds=idx * 5),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "direction": direction,
                "duration_sec": round(base_duration, 4),
                "packets": base_packets,
                "bytes": base_bytes,
                "tcp_syn_ratio": round(syn_ratio, 4),
                "tcp_ack_ratio": round(ack_ratio, 4),
                "tcp_fin_ratio": round(fin_ratio, 4),
                "inter_arrival_mean_ms": round(iat_mean, 4),
                "inter_arrival_std_ms": round(iat_std, 4),
                "failed_connections": failed_connections,
                "label": "anomalous" if anomalous else "normal",
            }
        )

    return pd.DataFrame(rows)


def generate_demo_dataset(
    output_dir: str | Path,
    days: int,
    rows_per_day: int,
    anomaly_fraction: float = 0.08,
    base_seed: int = 7,
) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for day in range(days):
        frame = generate_demo_flows(
            DemoConfig(
                rows=rows_per_day,
                anomaly_fraction=anomaly_fraction,
                seed=base_seed + day,
            )
        )
        frame["timestamp"] = pd.date_range(
            start=pd.Timestamp("2026-04-20") + pd.Timedelta(days=day),
            periods=len(frame),
            freq="5s",
        )
        file_path = destination / f"day{day + 1}.csv"
        frame.to_csv(file_path, index=False)
        created_files.append(file_path)

    return created_files


def load_flows_csv(path: str | Path) -> pd.DataFrame:
    candidate_encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error: Exception | None = None

    for encoding in candidate_encodings:
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            return prepare_flow_frame(frame)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            last_error = exc
            continue

    raise ValueError(f"Could not parse CSV file: {path}. Last error: {last_error}")


def load_kaggle_dataset(dataset_ref: str) -> tuple[pd.DataFrame, Path, list[Path]]:
    dataset_dir = _find_cached_kaggle_dataset_dir(dataset_ref)
    if dataset_dir is None:
        try:
            import kagglehub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "kagglehub is required for --kaggle-dataset. Install it with "
                "`python -m pip install kagglehub` in your active interpreter."
            ) from exc

        try:
            dataset_dir = Path(kagglehub.dataset_download(dataset_ref))
        except Exception as exc:
            raise ConnectionError(
                f"Could not download Kaggle dataset `{dataset_ref}` and no local cache was found. "
                "If you downloaded it before, check that it still exists under "
                "`%USERPROFILE%\\.cache\\kagglehub\\datasets`."
            ) from exc

    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found after downloading Kaggle dataset: {dataset_ref}"
        )
    return load_flow_sources(csv_files), dataset_dir, csv_files


def load_kaggle_datasets(dataset_refs: list[str]) -> tuple[pd.DataFrame, list[Path], list[Path]]:
    frames: list[pd.DataFrame] = []
    dataset_dirs: list[Path] = []
    csv_files: list[Path] = []

    for dataset_ref in dataset_refs:
        frame, dataset_dir, dataset_csvs = load_kaggle_dataset(dataset_ref)
        enriched = frame.copy()
        enriched["dataset_source"] = dataset_ref
        frames.append(enriched)
        dataset_dirs.append(dataset_dir)
        csv_files.extend(dataset_csvs)

    if not frames:
        raise FileNotFoundError("No Kaggle datasets were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    return prepare_flow_frame(combined), dataset_dirs, csv_files


def load_flow_sources(paths: list[str | Path]) -> pd.DataFrame:
    csv_files: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Input path does not exist: {path}. "
                "Pass a real CSV file, a folder containing CSVs, or multiple CSV paths."
            )

        if path.is_file():
            if path.suffix.lower() != ".csv":
                raise ValueError(f"Unsupported file type: {path}. Only .csv files are supported.")
            if _should_use_csv(path):
                csv_files.append(path)
            continue

        directory_csvs = [candidate for candidate in sorted(path.rglob("*.csv")) if _should_use_csv(candidate)]
        if not directory_csvs:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        csv_files.extend(directory_csvs)

    if not csv_files:
        raise FileNotFoundError("No CSV inputs were resolved.")

    frames: list[pd.DataFrame] = []
    skipped_files: list[tuple[Path, str]] = []
    for csv_path in csv_files:
        try:
            frames.append(load_flows_csv(csv_path))
        except ValueError as exc:
            skipped_files.append((csv_path, str(exc)))
            continue

    if skipped_files:
        print("[FlowMind] Skipped unreadable CSV files:")
        for csv_path, reason in skipped_files[:10]:
            print(f"  - {csv_path} ({reason})")
        if len(skipped_files) > 10:
            print(f"  - ... and {len(skipped_files) - 10} more")

    if not frames:
        raise FileNotFoundError("No readable CSV inputs were resolved.")

    combined = pd.concat(frames, ignore_index=True)
    return prepare_flow_frame(combined)


def limit_flow_rows(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame
    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return shuffled.head(max_rows).copy()


def prepare_flow_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _standardize_columns(frame.copy())

    if "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    else:
        prepared["timestamp"] = pd.date_range("2026-04-20", periods=len(prepared), freq="5s")

    prepared = _hydrate_flow_features(prepared)

    defaults = {
        "protocol": "TCP",
        "direction": "outbound",
        "src_ip": "10.0.0.10",
        "dst_ip": "8.8.8.8",
        "src_port": 40000,
        "dst_port": 443,
    }
    for column, default_value in defaults.items():
        if column not in prepared.columns:
            prepared[column] = default_value
        prepared[column] = prepared[column].fillna(default_value)

    if "direction" not in prepared.columns or prepared["direction"].isna().all():
        prepared["direction"] = prepared.apply(
            lambda row: _infer_direction(str(row.get("src_ip", "")), str(row.get("dst_ip", ""))),
            axis=1,
        )

    for column in NUMERIC_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = 0.0
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)

    prepared["packets"] = prepared["packets"].clip(lower=0)
    prepared["bytes"] = prepared["bytes"].clip(lower=0)
    prepared["duration_sec"] = prepared["duration_sec"].clip(lower=1e-6)
    prepared["inter_arrival_mean_ms"] = prepared["inter_arrival_mean_ms"].clip(lower=0, upper=60000)
    prepared["inter_arrival_std_ms"] = prepared["inter_arrival_std_ms"].clip(lower=0, upper=60000)

    prepared["protocol"] = prepared["protocol"].astype(str).str.upper().map(
        lambda value: PROTOCOL_NUMBER_MAP.get(value, value)
    )
    prepared["protocol"] = prepared["protocol"].apply(_canonicalize_protocol)
    prepared["direction"] = prepared["direction"].astype(str).str.lower()
    prepared["src_ip"] = prepared["src_ip"].astype(str)
    prepared["dst_ip"] = prepared["dst_ip"].astype(str)
    prepared["src_port"] = prepared["src_port"].astype(int)
    prepared["dst_port"] = prepared["dst_port"].astype(int)

    if "label" in prepared.columns:
        prepared["label"] = prepared["label"].apply(_normalize_label)

    return prepared.sort_values("timestamp").reset_index(drop=True)


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized_lookup = {
        _normalize_column_name(column): column for column in frame.columns
    }

    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            match = normalized_lookup.get(_normalize_column_name(alias))
            if match is not None:
                rename_map[match] = canonical
                break

    standardized = frame.rename(columns=rename_map)
    return standardized


def _hydrate_flow_features(frame: pd.DataFrame) -> pd.DataFrame:
    hydrated = frame.copy()

    if "packets" not in hydrated.columns:
        forward_packets = _numeric_series(hydrated, "total_fwd_packets")
        backward_packets = _numeric_series(hydrated, "total_backward_packets")
        if forward_packets is not None or backward_packets is not None:
            hydrated["packets"] = _coalesce_numeric(forward_packets, backward_packets)

    if "bytes" not in hydrated.columns:
        forward_bytes = _numeric_series(hydrated, "total_length_of_fwd_packets")
        backward_bytes = _numeric_series(hydrated, "total_length_of_bwd_packets")
        if forward_bytes is not None or backward_bytes is not None:
            hydrated["bytes"] = _coalesce_numeric(forward_bytes, backward_bytes)

    if "packets" not in hydrated.columns:
        source_packets = _numeric_series(hydrated, "src_pkts")
        destination_packets = _numeric_series(hydrated, "dst_pkts")
        if source_packets is not None or destination_packets is not None:
            hydrated["packets"] = _coalesce_numeric(source_packets, destination_packets)

    if "packets" not in hydrated.columns:
        source_packets = _numeric_series(hydrated, "spkts")
        destination_packets = _numeric_series(hydrated, "dpkts")
        if source_packets is not None or destination_packets is not None:
            hydrated["packets"] = _coalesce_numeric(source_packets, destination_packets)

    if "bytes" not in hydrated.columns:
        source_bytes = _numeric_series(hydrated, "src_bytes")
        destination_bytes = _numeric_series(hydrated, "dst_bytes")
        if source_bytes is not None or destination_bytes is not None:
            hydrated["bytes"] = _coalesce_numeric(source_bytes, destination_bytes)

    if "bytes" not in hydrated.columns:
        source_bytes = _numeric_series(hydrated, "sbytes")
        destination_bytes = _numeric_series(hydrated, "dbytes")
        if source_bytes is not None or destination_bytes is not None:
            hydrated["bytes"] = _coalesce_numeric(source_bytes, destination_bytes)

    if "duration_sec" in hydrated.columns:
        hydrated["duration_sec"] = _scale_microseconds_to_seconds(hydrated["duration_sec"])
    else:
        hydrated["duration_sec"] = pd.Series(0.0, index=hydrated.index)

    if "inter_arrival_mean_ms" in hydrated.columns:
        hydrated["inter_arrival_mean_ms"] = _scale_microseconds_to_milliseconds(hydrated["inter_arrival_mean_ms"])
    else:
        hydrated["inter_arrival_mean_ms"] = pd.Series(0.0, index=hydrated.index)

    if "inter_arrival_std_ms" in hydrated.columns:
        hydrated["inter_arrival_std_ms"] = _scale_microseconds_to_milliseconds(hydrated["inter_arrival_std_ms"])
    else:
        hydrated["inter_arrival_std_ms"] = pd.Series(0.0, index=hydrated.index)

    if "sinpkt" in hydrated.columns or "dinpkt" in hydrated.columns:
        sinpkt = _seconds_or_milliseconds_to_milliseconds(_numeric_series(hydrated, "sinpkt"), hydrated.index)
        dinpkt = _seconds_or_milliseconds_to_milliseconds(_numeric_series(hydrated, "dinpkt"), hydrated.index)
        hydrated["inter_arrival_mean_ms"] = _average_series(sinpkt, dinpkt, hydrated["inter_arrival_mean_ms"])

    if "sjit" in hydrated.columns or "djit" in hydrated.columns:
        sjit = _seconds_or_milliseconds_to_milliseconds(_numeric_series(hydrated, "sjit"), hydrated.index)
        djit = _seconds_or_milliseconds_to_milliseconds(_numeric_series(hydrated, "djit"), hydrated.index)
        hydrated["inter_arrival_std_ms"] = _average_series(sjit, djit, hydrated["inter_arrival_std_ms"])

    if "protocol" in hydrated.columns:
        hydrated["protocol"] = hydrated["protocol"].astype(str).str.upper()

    if "dst_port" not in hydrated.columns and "service" in hydrated.columns:
        hydrated["dst_port"] = hydrated["service"].astype(str).map(_service_name_to_port)

    if "failed_connections" not in hydrated.columns and "state" in hydrated.columns:
        state_text = hydrated["state"].astype(str).str.upper()
        hydrated["failed_connections"] = state_text.isin({"REJ", "RST", "RSTO", "RSTR", "S0", "SH", "OTH", "INT"}).astype(int)

    if "label" not in hydrated.columns and "attack_cat" in hydrated.columns:
        hydrated["label"] = hydrated["attack_cat"]

    packet_series = hydrated["packets"] if "packets" in hydrated.columns else pd.Series(0.0, index=hydrated.index)
    packet_base = pd.to_numeric(packet_series, errors="coerce").fillna(0).clip(lower=1)
    for ratio_column, count_column in (
        ("tcp_syn_ratio", "syn_flag_count"),
        ("tcp_ack_ratio", "ack_flag_count"),
        ("tcp_fin_ratio", "fin_flag_count"),
    ):
        if ratio_column not in hydrated.columns and count_column in hydrated.columns:
            counts = pd.to_numeric(hydrated[count_column], errors="coerce").fillna(0)
            hydrated[ratio_column] = counts / packet_base

    return hydrated


def _normalize_column_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _coalesce_numeric(first: pd.Series | None, second: pd.Series | None) -> pd.Series:
    if first is None and second is None:
        raise ValueError("At least one numeric series is required.")
    if first is None:
        return second.copy()  # type: ignore[union-attr]
    if second is None:
        return first.copy()
    return first.add(second, fill_value=0.0)


def _average_series(first: pd.Series | None, second: pd.Series | None, fallback: pd.Series) -> pd.Series:
    if first is None and second is None:
        return fallback
    if first is None:
        return second.copy() if second is not None else fallback
    if second is None:
        return first.copy()
    return (first + second) / 2.0


def _scale_microseconds_to_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if float(numeric.abs().median()) > 1000:
        return numeric / 1_000_000.0
    return numeric


def _scale_microseconds_to_milliseconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if float(numeric.abs().median()) > 1000:
        return numeric / 1_000.0
    return numeric


def _seconds_or_milliseconds_to_milliseconds(series: pd.Series | None, index: pd.Index) -> pd.Series | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    non_zero = numeric[numeric > 0]
    if non_zero.empty:
        return pd.Series(0.0, index=index)
    median_value = float(non_zero.median())
    if median_value < 10:
        return numeric * 1000.0
    return numeric


def _infer_direction(src_ip: str, dst_ip: str) -> str:
    try:
        src_private = ipaddress.ip_address(src_ip).is_private
        dst_private = ipaddress.ip_address(dst_ip).is_private
    except ValueError:
        return "outbound"

    if src_private and not dst_private:
        return "outbound"
    if not src_private and dst_private:
        return "inbound"
    return "outbound"


def _normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"0", "normal", "benign", "false", "legitimate"}:
        return "normal"
    if text in {"1", "attack", "anomaly", "anomalous", "malicious", "true"}:
        return "anomalous"
    return text or "unknown"


def _canonicalize_protocol(value: object) -> str:
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return "OTHER"
    if text in PROTOCOL_CANONICAL_MAP:
        return PROTOCOL_CANONICAL_MAP[text]
    if text.isdigit():
        return PROTOCOL_CANONICAL_MAP.get(PROTOCOL_NUMBER_MAP.get(text, text), "OTHER")
    if text.startswith("TCP"):
        return "TCP"
    if text.startswith("UDP"):
        return "UDP"
    if text.startswith("ICMP"):
        return "ICMP"
    return "OTHER"


def _should_use_csv(path: Path) -> bool:
    name = path.name.lower()
    return not any(pattern in name for pattern in SKIP_FILE_PATTERNS)


def _find_cached_kaggle_dataset_dir(dataset_ref: str) -> Path | None:
    parts = dataset_ref.split("/", maxsplit=1)
    if len(parts) != 2:
        return None

    owner, dataset_name = parts
    base_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / dataset_name
    versions_dir = base_dir / "versions"
    if not versions_dir.exists():
        return None

    version_dirs = [path for path in versions_dir.iterdir() if path.is_dir()]
    if not version_dirs:
        return None

    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)$", path.name)
        if match:
            return (int(match.group(1)), path.name)
        return (-1, path.name)

    return sorted(version_dirs, key=sort_key)[-1]


def _service_name_to_port(value: str) -> int:
    service = value.strip().lower()
    service_ports = {
        "http": 80,
        "https": 443,
        "dns": 53,
        "ssh": 22,
        "ftp": 21,
        "ftp-data": 20,
        "smtp": 25,
        "pop3": 110,
        "imap": 143,
        "dhcp": 67,
        "ntp": 123,
        "irc": 194,
        "ssl": 443,
        "mqtt": 1883,
        "telnet": 23,
    }
    return service_ports.get(service, 0)
