from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class PeakEstimate:
    """Peak pressure estimate for a serving instance."""

    instance_id: int
    peak_tokens: float
    utilization: float
    should_scale_up: bool
    should_scale_down: bool


class PeakWorkloadAnticipator:
    """PRISM peak-workload anticipator for adaptive scaling.

    The anticipator projects near-term memory pressure from serving-layer
    counters and lookahead state. It exposes the decision signal used by PRISM's
    adaptive tuning stage without changing the underlying model server.
    """

    def __init__(
        self,
        peak_memory_capacity_tokens: float = 8192.0,
        scale_up_memory_threshold: float = 0.85,
        scale_down_memory_threshold: float = 0.20,
        scaling_violation_tolerance: float = 0.20,
    ):
        if peak_memory_capacity_tokens <= 0:
            raise ValueError("peak_memory_capacity_tokens must be positive")
        if not 0 <= scale_down_memory_threshold <= scale_up_memory_threshold <= 1:
            raise ValueError("memory thresholds must satisfy 0 <= down <= up <= 1")
        if not 0 <= scaling_violation_tolerance <= 1:
            raise ValueError("scaling_violation_tolerance must be in [0, 1]")
        self.peak_memory_capacity_tokens = float(peak_memory_capacity_tokens)
        self.scale_up_memory_threshold = float(scale_up_memory_threshold)
        self.scale_down_memory_threshold = float(scale_down_memory_threshold)
        self.scaling_violation_tolerance = float(scaling_violation_tolerance)

    def estimate_instance(self, instance: object) -> PeakEstimate:
        """Estimate peak pressure for one active instance."""

        current_tokens = (
            float(instance.get_instance_incoming_prefill_tokens())
            + float(instance.get_instance_incoming_decode_tokens())
        )
        expected_tokens = float(instance.get_instance_expected_token_usage())
        lookahead_tokens = float(instance.get_instance_lookahead_max_tokens())
        peak_tokens = max(current_tokens, expected_tokens, lookahead_tokens)
        utilization = peak_tokens / self.peak_memory_capacity_tokens
        return PeakEstimate(
            instance_id=int(instance.get_instance_id()),
            peak_tokens=peak_tokens,
            utilization=utilization,
            should_scale_up=utilization > self.scale_up_memory_threshold,
            should_scale_down=utilization < self.scale_down_memory_threshold,
        )

    def estimate(self, instances: Iterable[object]) -> List[PeakEstimate]:
        """Estimate peak pressure for all active instances."""

        return [self.estimate_instance(instance) for instance in instances]

    def desired_instances(self, instances: Iterable[object], current_num_instances: int) -> int:
        """Return the adaptive scaling target implied by peak pressure."""

        estimates = self.estimate(instances)
        if not estimates:
            return current_num_instances

        high_ratio = sum(1 for estimate in estimates if estimate.should_scale_up) / len(estimates)
        low_ratio = sum(1 for estimate in estimates if estimate.should_scale_down) / len(estimates)

        if high_ratio > self.scaling_violation_tolerance:
            return current_num_instances + 1
        if low_ratio > 1.0 - self.scaling_violation_tolerance:
            return current_num_instances - 1
        return current_num_instances


class PrismScaler:
    """Asynchronous adaptive scaler for PRISM runtime tuning."""

    def __init__(
        self,
        scheduler: object,
        anticipator: PeakWorkloadAnticipator,
        interval: float = 10.0,
        cold_start_time: float = 0.0,
        scale_freeze_time: float = 0.0,
    ):
        if interval <= 0:
            raise ValueError("interval must be positive")
        if cold_start_time < 0 or scale_freeze_time < 0:
            raise ValueError("scaling delays must be non-negative")
        self.scheduler = scheduler
        self.anticipator = anticipator
        self.interval = float(interval)
        self.cold_start_time = float(cold_start_time)
        self.scale_freeze_time = float(scale_freeze_time)
        self.monitor_task = None
        self.monitor_stop_event = asyncio.Event()
        self.scaling_lock = asyncio.Lock()

    async def monitor_start(self) -> None:
        """Start the adaptive tuning loop."""

        self.monitor_task = asyncio.create_task(self._monitor_loop())

    async def monitor_stop(self) -> None:
        """Stop the adaptive tuning loop and wait for completion."""

        if self.monitor_task:
            self.monitor_stop_event.set()
            await self.monitor_task

    async def _monitor_loop(self) -> None:
        while not self.monitor_stop_event.is_set():
            try:
                await asyncio.wait_for(self.monitor_stop_event.wait(), timeout=self.interval)
                break
            except asyncio.TimeoutError:
                pass
            desired = self.anticipator.desired_instances(
                self.scheduler.instances,
                self.scheduler.num_instances,
            )
            if desired != self.scheduler.num_instances:
                asyncio.create_task(self.scale_to(desired))

    async def scale_to(self, num_instances: int) -> None:
        """Scale the scheduler to a bounded target instance count."""

        num_instances = self.scheduler.clamp_num_instances(num_instances)
        if num_instances == self.scheduler.num_instances:
            return
        if self.scaling_lock.locked():
            return

        async with self.scaling_lock:
            await asyncio.sleep(self.cold_start_time)
            self.scheduler.scale_instances_to(num_instances)
            await asyncio.sleep(self.scale_freeze_time)
