from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict


CONFIG_ALIASES = {
    "history_size": "response_history_size",
    "decay": "response_recency_decay",
    "sigma": "prompt_similarity_sigma",
    "dirichlet_alpha": "response_length_prior",
    "default_output_len": "fallback_response_len",
    "prediction_mode": "response_prediction_mode",
    "prefill_alpha": "routing_prefill_alpha",
    "decode_beta": "routing_decode_beta",
    "heterogeneity_weight": "routing_heterogeneity_weight",
    "congestion_weight": "routing_congestion_weight",
    "memory_capacity_tokens": "peak_memory_capacity_tokens",
    "memory_high_threshold": "scale_up_memory_threshold",
    "memory_low_threshold": "scale_down_memory_threshold",
    "acceptable_violation_rate": "scaling_violation_tolerance",
    "scaling_interval": "adaptive_scaling_interval",
    "cold_start_time": "instance_cold_start_seconds",
    "scale_freeze_time": "scaling_cooldown_seconds",
    "approximate_features": "implementation_notes",
}


@dataclass
class PrismConfig:
    """Configuration for the PRISM research prototype.

    Defaults prioritize reproducible behavior across local environments. Model
    and system constants can be tuned per deployment through ``prism_config``.
    """

    response_history_size: int = 128
    response_recency_decay: float = 0.96
    prompt_similarity_sigma: float = 256.0
    response_length_prior: float = 1.0
    fallback_response_len: int = 256
    response_prediction_mode: str = "expectation"
    random_seed: int = 7

    routing_prefill_alpha: float = 1e-6
    routing_decode_beta: float = 1e-3
    routing_heterogeneity_weight: float = 1.0
    routing_congestion_weight: float = 1.0

    slo_seconds: float = 30.0
    per_token_memory_bytes: float = 1.0
    peak_memory_capacity_tokens: float = 8192.0
    scale_up_memory_threshold: float = 0.85
    scale_down_memory_threshold: float = 0.20
    scaling_violation_tolerance: float = 0.20
    adaptive_scaling_interval: float = 10.0
    instance_cold_start_seconds: float = 0.0
    scaling_cooldown_seconds: float = 0.0

    implementation_notes: Dict[str, str] = field(default_factory=lambda: {
        "intra_instance_preemption": (
            "The current artifact records PRISM pressure scores and routing "
            "state at the framework layer. Direct vLLM iteration-level "
            "preemption requires scheduler hooks below the OpenAI-compatible "
            "HTTP interface."
        ),
        "peak_memory": (
            "Peak pressure is estimated from framework-visible request "
            "counters and lookahead state, matching the public serving API."
        ),
        "workload_forecast": (
            "This artifact emphasizes online scheduling and adaptive tuning; "
            "periodic planning can be supplied through the same scaler API."
        ),
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.response_history_size <= 0:
            raise ValueError("response_history_size must be positive")
        if not 0 < self.response_recency_decay <= 1:
            raise ValueError("response_recency_decay must be in (0, 1]")
        if self.prompt_similarity_sigma <= 0:
            raise ValueError("prompt_similarity_sigma must be positive")
        if self.response_length_prior <= 0:
            raise ValueError("response_length_prior must be positive")
        if self.fallback_response_len <= 0:
            raise ValueError("fallback_response_len must be positive")
        if self.response_prediction_mode not in {"expectation", "sample"}:
            raise ValueError("response_prediction_mode must be 'expectation' or 'sample'")
        if self.peak_memory_capacity_tokens <= 0:
            raise ValueError("peak_memory_capacity_tokens must be positive")
        for name in ("scale_up_memory_threshold", "scale_down_memory_threshold", "scaling_violation_tolerance"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.scale_down_memory_threshold > self.scale_up_memory_threshold:
            raise ValueError("scale_down_memory_threshold cannot exceed scale_up_memory_threshold")
        if self.adaptive_scaling_interval <= 0:
            raise ValueError("adaptive_scaling_interval must be positive")
        if self.instance_cold_start_seconds < 0 or self.scaling_cooldown_seconds < 0:
            raise ValueError("scaling delays must be non-negative")


def load_prism_config(path: str | None = None) -> PrismConfig:
    """Load a PRISM configuration file.

    Older configuration keys are accepted and normalized to the current naming
    scheme, which keeps existing scripts reproducible while presenting a clean
    configuration surface for new experiments.
    """

    if path is None:
        return PrismConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw = _normalize_config_keys(raw)
    defaults = PrismConfig().to_dict()
    defaults.update(raw)
    try:
        return PrismConfig(**defaults)
    except TypeError as exc:
        raise ValueError(f"Invalid PRISM configuration keys in {path}: {exc}") from exc


def save_default_prism_config(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(PrismConfig().to_dict(), f, indent=4)


def _normalize_config_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(raw)
    for old_key, new_key in CONFIG_ALIASES.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(old_key)
        elif old_key in normalized:
            normalized.pop(old_key)
    return normalized
