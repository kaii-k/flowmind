from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import prepare_flow_frame


SERVICE_PORTS = {
    20: "ftp-data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    123: "ntp",
    443: "https",
    445: "smb",
    3389: "rdp",
    8080: "http-alt",
}

FEATURE_LABELS = {
    "log_duration": "session duration",
    "log_packets": "packet volume",
    "log_bytes": "byte volume",
    "bytes_per_packet": "bytes per packet",
    "bytes_per_sec": "throughput",
    "packets_per_sec": "packet rate",
    "tcp_syn_ratio": "SYN ratio",
    "tcp_ack_ratio": "ACK ratio",
    "tcp_fin_ratio": "FIN ratio",
    "inter_arrival_mean_ms": "mean inter-arrival",
    "inter_arrival_std_ms": "inter-arrival jitter",
    "failed_connections": "failed connections",
    "is_inbound": "inbound direction",
    "is_tcp": "TCP usage",
    "is_udp": "UDP usage",
    "is_icmp": "ICMP usage",
    "is_other_proto": "other-protocol usage",
    "is_web_service": "web-service traffic",
    "is_admin_service": "admin-service traffic",
    "is_uncommon_port": "uncommon destination port",
}

EMBEDDING_COLUMNS = [
    "log_duration",
    "log_packets",
    "log_bytes",
    "bytes_per_packet",
    "bytes_per_sec",
    "packets_per_sec",
    "tcp_syn_ratio",
    "tcp_ack_ratio",
    "tcp_fin_ratio",
    "log_inter_arrival_mean_ms",
    "log_inter_arrival_std_ms",
    "failed_connections",
    "is_inbound",
    "is_tcp",
    "is_udp",
    "is_icmp",
    "is_other_proto",
    "is_web_service",
    "is_admin_service",
    "is_uncommon_port",
]


@dataclass
class ScoreArtifacts:
    scored: pd.DataFrame
    threshold: float
    summary: dict[str, Any]


@dataclass
class AdaptiveArtifacts:
    scored: pd.DataFrame
    threshold: float
    summary: dict[str, Any]
    batch_summaries: pd.DataFrame


class FlowMindAI:
    def __init__(self, contamination: float = 0.08) -> None:
        self.contamination = float(np.clip(contamination, 0.01, 0.35))
        self.medians: pd.Series | None = None
        self.mads: pd.Series | None = None
        self.components: np.ndarray | None = None
        self.mean_vector: np.ndarray | None = None
        self.context_frequency: dict[str, float] = {}
        self.threshold_: float | None = None
        self.training_scores_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "FlowMindAI":
        prepared = prepare_flow_frame(frame)
        embedding = self._build_embedding(prepared)

        self.medians = embedding.median()
        mad = (embedding - self.medians).abs().median()
        self.mads = mad.replace(0, 1e-6)

        scaled = self._robust_scale(embedding)
        self.mean_vector = scaled.mean(axis=0).to_numpy()

        centered = scaled.to_numpy() - self.mean_vector
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        component_count = int(min(4, vt.shape[0], vt.shape[1]))
        self.components = vt[:component_count]

        context_keys = self._context_keys(prepared)
        context_freq = pd.Series(context_keys).value_counts(normalize=True)
        self.context_frequency = context_freq.to_dict()

        training_scores = self._score_from_scaled(scaled, context_keys)
        self.training_scores_ = training_scores
        self.threshold_ = self._compute_decision_threshold(training_scores, prepared)
        return self

    def score(self, frame: pd.DataFrame) -> ScoreArtifacts:
        if self.medians is None or self.mads is None or self.components is None or self.mean_vector is None:
            raise RuntimeError("Model must be fit before scoring.")

        prepared = prepare_flow_frame(frame)
        embedding = self._build_embedding(prepared)
        scaled = self._robust_scale(embedding)
        context_keys = self._context_keys(prepared)
        anomaly_score = self._score_from_scaled(scaled, context_keys)
        threshold = self._compute_decision_threshold(
            anomaly_score,
            prepared,
            baseline_threshold=self.threshold_,
        )

        scored = prepared.copy()
        scored["service"] = scored["dst_port"].map(SERVICE_PORTS).fillna("other")
        scored["anomaly_score"] = anomaly_score
        scored["is_anomaly"] = scored["anomaly_score"] >= threshold
        scored["risk_level"] = scored["anomaly_score"].apply(self._risk_label)
        scored["explanation"] = self._explain(embedding, scaled, scored)

        summary = {
            "total_flows": int(len(scored)),
            "flagged_flows": int(scored["is_anomaly"].sum()),
            "flag_rate": round(float(scored["is_anomaly"].mean() * 100), 2),
            "mean_score": round(float(scored["anomaly_score"].mean()), 4),
            "max_score": round(float(scored["anomaly_score"].max()), 4),
            "top_protocol": str(scored["protocol"].mode().iloc[0]) if not scored.empty else "N/A",
        }
        if "label" in scored.columns:
            known_attack_mask = self._known_attack_mask(scored)
            known_normal_mask = self._known_normal_mask(scored)
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
        return ScoreArtifacts(scored=scored, threshold=threshold, summary=summary)

    def fit_score(self, frame: pd.DataFrame) -> ScoreArtifacts:
        return self.fit(frame).score(frame)

    def adaptive_fit_score(
        self,
        frame: pd.DataFrame,
        batch_size: int = 1000,
        warmup_batches: int = 1,
        memory_size: int | None = None,
        progress_every: int = 25,
    ) -> AdaptiveArtifacts:
        prepared = prepare_flow_frame(frame)
        batch_size = max(100, batch_size)
        warmup_batches = max(1, warmup_batches)
        memory_size = memory_size or batch_size * 5
        progress_every = max(1, progress_every)
        min_relearn_rows = max(100, batch_size // 20)

        batches = [
            prepared.iloc[start:start + batch_size].copy()
            for start in range(0, len(prepared), batch_size)
        ]
        if not batches:
            raise ValueError("No flows available for adaptive learning.")

        warmup_count = min(warmup_batches, len(batches))
        warmup_frame = pd.concat(batches[:warmup_count], ignore_index=True)
        warmup_fit_frame = self._select_benign_reference(warmup_frame, allow_full_fallback=True)
        self.fit(warmup_fit_frame)

        scored_batches: list[pd.DataFrame] = []
        summary_rows: list[dict[str, Any]] = []

        warmup_result = self.score(warmup_frame)
        warmup_scored = warmup_result.scored.copy()
        warmup_scored["batch_id"] = 1
        warmup_scored["learning_phase"] = "warmup"
        scored_batches.append(warmup_scored)
        summary_rows.append(
            self._batch_summary_row(
                batch_id=1,
                phase="warmup",
                size=len(warmup_frame),
                threshold=warmup_result.threshold,
                scored=warmup_scored,
                relearn_status="warmup",
            )
        )

        memory_frame = warmup_fit_frame.tail(memory_size).copy()
        next_batch_id = 2
        total_batches = len(batches)
        skipped_relearn_batches = 0
        print(
            f"[FlowMind] Warmup complete: batch 1/{total_batches}, "
            f"threshold={warmup_result.threshold:.4f}, "
            f"flagged={int(warmup_scored['is_anomaly'].sum())}/{len(warmup_scored)}"
        )

        for batch in batches[warmup_count:]:
            result = self.score(batch)
            scored_batch = result.scored.copy()
            scored_batch["batch_id"] = next_batch_id
            scored_batch["learning_phase"] = "adapt"
            scored_batches.append(scored_batch)
            summary_rows.append(
                self._batch_summary_row(
                    batch_id=next_batch_id,
                    phase="adapt",
                    size=len(batch),
                    threshold=result.threshold,
                    scored=scored_batch,
                    relearn_status="updated",
                )
            )

            candidate_normals = self._select_benign_reference(result.scored, allow_full_fallback=False)
            has_labels = "label" in scored_batch.columns
            should_relearn = True

            if candidate_normals.empty:
                candidate_normals = scored_batch.loc[~result.scored["is_anomaly"].to_numpy()].copy()
            if has_labels and len(candidate_normals) < min_relearn_rows:
                should_relearn = False
            elif candidate_normals.empty:
                candidate_normals = scored_batch.nsmallest(
                    max(1, len(scored_batch) // 3),
                    "anomaly_score",
                ).drop(
                    columns=["service", "anomaly_score", "is_anomaly", "risk_level", "explanation", "batch_id", "learning_phase"],
                    errors="ignore",
                )

            if should_relearn and not candidate_normals.empty:
                memory_frame = pd.concat([memory_frame, candidate_normals], ignore_index=True).tail(memory_size)
                self.fit(memory_frame)
            else:
                skipped_relearn_batches += 1
                summary_rows[-1]["relearn_status"] = "skipped"
            if next_batch_id == total_batches or next_batch_id % progress_every == 0:
                status_text = "relearn=skipped" if not should_relearn or candidate_normals.empty else "relearn=updated"
                print(
                    f"[FlowMind] Processed batch {next_batch_id}/{total_batches} "
                    f"flagged={int(scored_batch['is_anomaly'].sum())}/{len(scored_batch)} "
                    f"threshold={result.threshold:.4f} {status_text}"
                )
            next_batch_id += 1

        scored = pd.concat(scored_batches, ignore_index=True)
        summary = {
            "total_flows": int(len(scored)),
            "flagged_flows": int(scored["is_anomaly"].sum()),
            "flag_rate": round(float(scored["is_anomaly"].mean() * 100), 2),
            "mean_score": round(float(scored["anomaly_score"].mean()), 4),
            "max_score": round(float(scored["anomaly_score"].max()), 4),
            "top_protocol": str(scored["protocol"].mode().iloc[0]) if not scored.empty else "N/A",
            "adaptive_batches": int(len(summary_rows)),
            "warmup_batches": int(warmup_count),
            "memory_size": int(memory_size),
            "skipped_relearn_batches": int(skipped_relearn_batches),
        }
        if "label" in scored.columns:
            known_attack_mask = self._known_attack_mask(scored)
            known_normal_mask = self._known_normal_mask(scored)
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

        final_scoring_threshold = float(summary_rows[-1]["threshold"]) if summary_rows else float(self.threshold_ or 0.0)

        return AdaptiveArtifacts(
            scored=scored,
            threshold=final_scoring_threshold,
            summary=summary,
            batch_summaries=pd.DataFrame(summary_rows),
        )

    def _build_embedding(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(index=frame.index)

        duration = frame["duration_sec"].clip(lower=0.001)
        packets = frame["packets"].clip(lower=1)
        bytes_ = frame["bytes"].clip(lower=1)

        data["log_duration"] = np.log1p(duration)
        data["log_packets"] = np.log1p(packets)
        data["log_bytes"] = np.log1p(bytes_)
        data["bytes_per_packet"] = bytes_ / packets
        data["bytes_per_sec"] = np.log1p(bytes_ / duration)
        data["packets_per_sec"] = np.log1p(packets / duration)
        data["tcp_syn_ratio"] = frame["tcp_syn_ratio"].clip(0, 1)
        data["tcp_ack_ratio"] = frame["tcp_ack_ratio"].clip(0, 1)
        data["tcp_fin_ratio"] = frame["tcp_fin_ratio"].clip(0, 1)
        data["log_inter_arrival_mean_ms"] = np.log1p(frame["inter_arrival_mean_ms"].clip(lower=0))
        data["log_inter_arrival_std_ms"] = np.log1p(frame["inter_arrival_std_ms"].clip(lower=0))
        data["failed_connections"] = frame["failed_connections"].clip(lower=0)
        data["is_inbound"] = (frame["direction"] == "inbound").astype(float)
        data["is_tcp"] = (frame["protocol"] == "TCP").astype(float)
        data["is_udp"] = (frame["protocol"] == "UDP").astype(float)
        data["is_icmp"] = (frame["protocol"] == "ICMP").astype(float)
        data["is_other_proto"] = (~frame["protocol"].isin(["TCP", "UDP", "ICMP"])).astype(float)
        data["is_web_service"] = frame["dst_port"].isin([80, 443, 8080]).astype(float)
        data["is_admin_service"] = frame["dst_port"].isin([22, 23, 445, 3389]).astype(float)
        data["is_uncommon_port"] = (~frame["dst_port"].isin(list(SERVICE_PORTS.keys()))).astype(float)

        return data[EMBEDDING_COLUMNS].astype(float)

    def _robust_scale(self, embedding: pd.DataFrame) -> pd.DataFrame:
        if self.medians is None or self.mads is None:
            raise RuntimeError("Robust scaling parameters are not initialized.")
        return (embedding - self.medians) / self.mads

    def _context_keys(self, frame: pd.DataFrame) -> pd.Series:
        port_band = frame["dst_port"].apply(self._port_band)
        return frame["protocol"].astype(str) + "|" + frame["direction"].astype(str) + "|" + port_band.astype(str)

    @staticmethod
    def _port_band(port: int) -> str:
        if port in SERVICE_PORTS:
            return SERVICE_PORTS[port]
        if port < 1024:
            return "system-other"
        if port < 10000:
            return "registered"
        return "dynamic"

    def _score_from_scaled(self, scaled: pd.DataFrame, context_keys: pd.Series) -> np.ndarray:
        if self.components is None or self.mean_vector is None:
            raise RuntimeError("Model components are not initialized.")

        values = scaled.to_numpy()
        centered = np.clip(values - self.mean_vector, -12.0, 12.0)

        distance_score = np.sqrt(np.mean(centered**2, axis=1))

        projected = centered @ self.components.T
        reconstructed = projected @ self.components
        reconstruction_error = np.sqrt(np.mean((centered - reconstructed) ** 2, axis=1))

        rarity = np.array(
            [-np.log(self.context_frequency.get(key, 0.0001)) for key in context_keys],
            dtype=float,
        )

        score = (1.6 * distance_score) + (1.2 * reconstruction_error) + (0.85 * rarity)
        return score

    def _compute_decision_threshold(
        self,
        scores: np.ndarray,
        frame: pd.DataFrame,
        baseline_threshold: float | None = None,
    ) -> float:
        threshold_candidates = []
        if baseline_threshold is not None:
            threshold_candidates.append(float(baseline_threshold))

        scores_array = np.asarray(scores, dtype=float)
        default_quantile = float(np.quantile(scores_array, 1 - self.contamination))
        threshold_candidates.append(default_quantile)

        if "label" in frame.columns:
            benign_mask = self._known_normal_mask(frame).to_numpy()
            benign_scores = scores_array[benign_mask]
            if benign_scores.size > 0:
                benign_quantile = float(np.quantile(benign_scores, 0.995))
                benign_median = float(np.median(benign_scores))
                benign_mad = float(np.median(np.abs(benign_scores - benign_median)))
                robust_cutoff = benign_median + max(3.5 * benign_mad, 1.0)
                threshold_candidates.extend([benign_quantile, robust_cutoff])

        return max(threshold_candidates)

    def _explain(
        self,
        embedding: pd.DataFrame,
        scaled: pd.DataFrame,
        scored: pd.DataFrame,
    ) -> list[str]:
        explanations: list[str] = []
        for idx in range(len(scored)):
            row = embedding.iloc[idx]
            scaled_row = scaled.iloc[idx].abs().sort_values(ascending=False)
            top_features = scaled_row.head(3).index.tolist()
            parts = []
            for feature in top_features:
                human_value = row[feature]
                label = FEATURE_LABELS.get(feature, feature)
                parts.append(f"{label}={human_value:.2f}")

            port = int(scored.iloc[idx]["dst_port"])
            protocol = scored.iloc[idx]["protocol"]
            direction = scored.iloc[idx]["direction"]
            context = f"context={protocol}/{direction}/port-{port}"
            explanations.append(", ".join(parts + [context]))
        return explanations

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= 18:
            return "critical"
        if score >= 12:
            return "high"
        if score >= 7:
            return "medium"
        return "low"

    @staticmethod
    def _batch_summary_row(
        batch_id: int,
        phase: str,
        size: int,
        threshold: float,
        scored: pd.DataFrame,
        relearn_status: str,
    ) -> dict[str, Any]:
        row = {
            "batch_id": batch_id,
            "phase": phase,
            "batch_size": int(size),
            "threshold": round(float(threshold), 4),
            "flagged_flows": int(scored["is_anomaly"].sum()),
            "flag_rate": round(float(scored["is_anomaly"].mean() * 100), 2),
            "mean_score": round(float(scored["anomaly_score"].mean()), 4),
            "max_score": round(float(scored["anomaly_score"].max()), 4),
            "relearn_status": relearn_status,
        }
        if "label" in scored.columns:
            known_attack_mask = FlowMindAI._known_attack_mask(scored)
            known_normal_mask = FlowMindAI._known_normal_mask(scored)
            row["known_attack_flows"] = int(known_attack_mask.sum())
            row["known_normal_flows"] = int(known_normal_mask.sum())
            if int(known_attack_mask.sum()) > 0:
                row["detected_known_attacks"] = int((scored["is_anomaly"] & known_attack_mask).sum())
            if int(known_normal_mask.sum()) > 0:
                benign_flagged = int((scored["is_anomaly"] & known_normal_mask).sum())
                row["benign_false_positives"] = benign_flagged
                row["benign_false_positive_rate"] = round((benign_flagged / int(known_normal_mask.sum())) * 100, 2)
        return row

    @staticmethod
    def _known_attack_mask(frame: pd.DataFrame) -> pd.Series:
        return ~frame["label"].astype(str).isin(["normal", "benign", "0", "unknown"])

    @staticmethod
    def _known_normal_mask(frame: pd.DataFrame) -> pd.Series:
        return frame["label"].astype(str).isin(["normal", "benign", "0"])

    def _select_benign_reference(self, frame: pd.DataFrame, allow_full_fallback: bool = False) -> pd.DataFrame:
        if "label" in frame.columns:
            benign = frame.loc[self._known_normal_mask(frame)].copy()
            if len(benign) >= 250:
                return benign
            if not benign.empty and "anomaly_score" in frame.columns:
                supplement_count = max(250 - len(benign), 0)
                supplement = frame.loc[~self._known_attack_mask(frame)].nsmallest(
                    supplement_count,
                    "anomaly_score",
                )
                combined = pd.concat([benign, supplement], ignore_index=True).drop_duplicates()
                if not combined.empty:
                    return combined
            return frame.copy() if allow_full_fallback else frame.iloc[0:0].copy()
        return frame.copy()
