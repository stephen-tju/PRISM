from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RoutingDecision:
    """Complete cost breakdown for one PRISM routing decision."""

    instance: object
    instance_id: int
    cost: float
    prefill_latency: float
    decode_latency: float
    heterogeneity: float
    congestion: float


class PrismRouter:
    """Heterogeneity-aware inter-instance router.

    The router estimates per-instance execution cost from prompt prefill cost,
    decode cost, prompt-length heterogeneity, and current utilization. It keeps
    the decision local and online: each request is scored against the current
    candidate instances and assigned to the lowest-cost instance.
    """

    def __init__(
        self,
        routing_prefill_alpha: float = 1e-6,
        routing_decode_beta: float = 1e-3,
        routing_heterogeneity_weight: float = 1.0,
        routing_congestion_weight: float = 1.0,
    ):
        self.routing_prefill_alpha = float(routing_prefill_alpha)
        self.routing_decode_beta = float(routing_decode_beta)
        self.routing_heterogeneity_weight = float(routing_heterogeneity_weight)
        self.routing_congestion_weight = float(routing_congestion_weight)

    def choose_instance(
        self,
        instances: Iterable[object],
        prompt_len: int,
        predicted_output_len: int,
    ) -> RoutingDecision:
        """Return the best instance under the PRISM routing objective."""

        decisions = [
            self.score_instance(instance, prompt_len, predicted_output_len)
            for instance in instances
        ]
        if not decisions:
            raise ValueError("PRISM router received no candidate instances")
        return min(decisions, key=lambda decision: (decision.cost, decision.instance_id))

    def score_instance(
        self,
        instance: object,
        prompt_len: int,
        predicted_output_len: int,
    ) -> RoutingDecision:
        """Score a candidate instance for a single incoming request."""

        prompt_len = max(1, int(prompt_len))
        predicted_output_len = max(1, int(predicted_output_len))
        avg_prompt_len = self._avg_prompt_len(instance, prompt_len)
        heterogeneity = abs(prompt_len - avg_prompt_len) / max(1.0, prompt_len + avg_prompt_len)
        heterogeneity *= self.routing_heterogeneity_weight

        prefill_latency = self.routing_prefill_alpha * (prompt_len ** 2) * (1.0 + heterogeneity)
        decode_latency = self.routing_decode_beta * (prompt_len + predicted_output_len)

        utilization = self._utilization(instance)
        congestion = 1.0 + self.routing_congestion_weight * utilization
        cost = (prefill_latency + decode_latency) * congestion

        return RoutingDecision(
            instance=instance,
            instance_id=int(instance.get_instance_id()),
            cost=float(cost),
            prefill_latency=float(prefill_latency),
            decode_latency=float(decode_latency),
            heterogeneity=float(heterogeneity),
            congestion=float(congestion),
        )

    @staticmethod
    def _avg_prompt_len(instance: object, fallback_prompt_len: int) -> float:
        request_num = int(instance.get_instance_request_num())
        if request_num <= 0:
            return float(fallback_prompt_len)
        return float(instance.get_instance_incoming_prefill_tokens()) / request_num

    @staticmethod
    def _utilization(instance: object) -> float:
        gpu = float(getattr(instance, "gpu_utilization", 0.0))
        mem = float(getattr(instance, "gpu_memory_utilization", 0.0))
        # Framework metrics may expose GPU utilization as 0..100 and cache
        # utilization as 0..1; normalize before combining them.
        if gpu > 1.0:
            gpu = gpu / 100.0
        return max(0.0, min(1.0, max(gpu, mem)))


def pressure_score(predicted_output_len: int, decoded_len: int, elapsed_seconds: float) -> float:
    """Compute the PRISM pressure score for iteration-level scheduling."""

    remaining = max(0, int(predicted_output_len) - int(decoded_len))
    elapsed_seconds = max(1e-6, float(elapsed_seconds))
    return 1.0 - pow(2.718281828459045, -(remaining / elapsed_seconds))
