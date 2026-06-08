from __future__ import annotations

import copy
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Sequence

from .config import PrismConfig
from .response_perceptron import ResponseLengthPerceptron
from .routing import PrismRouter


logger = logging.getLogger(__name__)


class PrismScheduler:
    """PRISM scheduler coordinator.

    The scheduler owns response-length perception and inter-instance routing
    while reusing the established instance execution path. This keeps PRISM
    integrated with the existing serving foundation without changing baseline
    entrypoints.
    """

    def __init__(
        self,
        scheduler_config: Dict[str, Any],
        instance_config: Dict[str, Any],
        prism_config: PrismConfig | None = None,
        dry_run: bool = False,
    ):
        self.prism_config = prism_config or PrismConfig()
        self.dry_run = dry_run
        self.num_instances = int(scheduler_config["num_instances"])
        if self.num_instances < 1:
            raise ValueError("num_instances must be at least 1")

        instance_slots = self._load_instance_configuration(scheduler_config["instance_configurations"])
        if not instance_slots:
            raise ValueError("instance_configurations must contain at least one instance slot")
        if len(instance_slots) < self.num_instances:
            raise ValueError("The initial number of instances exceeded available slots.")

        self.instance_slots: List[object] = []
        instance_cls = SimulationInstance if dry_run else self._load_preserve_instance_cls()
        for ins_id, ins_cfg in enumerate(instance_slots):
            detailed_config = copy.deepcopy(instance_config)
            detailed_config.update(ins_cfg)
            self.instance_slots.append(instance_cls(ins_id, detailed_config, scheduler_config))
        self.instances = self.instance_slots[: self.num_instances]

        self.max_model_len = max(instance_config["max_model_len"], instance_config["max_num_seqs"])
        self.perceptron = ResponseLengthPerceptron(
            response_history_size=self.prism_config.response_history_size,
            response_recency_decay=self.prism_config.response_recency_decay,
            prompt_similarity_sigma=self.prism_config.prompt_similarity_sigma,
            response_length_prior=self.prism_config.response_length_prior,
            fallback_response_len=self.prism_config.fallback_response_len,
            response_prediction_mode=self.prism_config.response_prediction_mode,
            random_seed=self.prism_config.random_seed,
        )
        self.router = PrismRouter(
            routing_prefill_alpha=self.prism_config.routing_prefill_alpha,
            routing_decode_beta=self.prism_config.routing_decode_beta,
            routing_heterogeneity_weight=self.prism_config.routing_heterogeneity_weight,
            routing_congestion_weight=self.prism_config.routing_congestion_weight,
        )
        self.prompt_tokens_acc = 0
        self.response_tokens_acc = 0
        logger.info("PRISM scheduler initialized with %d/%d instances.", self.num_instances, len(self.instance_slots))

    @staticmethod
    def _load_instance_configuration(instance_config_file: str) -> list[dict[str, Any]]:
        """Load serving instance slots from the shared instance table."""

        if not instance_config_file:
            raise ValueError("instance_configurations path is required")
        with open(instance_config_file, "r", encoding="utf-8") as f:
            instance_slots = json.load(f)
        if not isinstance(instance_slots, list):
            raise ValueError("instance_configurations must be a JSON list")
        return instance_slots

    @staticmethod
    def _load_preserve_instance_cls():
        """Load the production instance class only for live serving runs."""

        from LLMServe.serve_instance.instance import Instance

        return Instance

    def get_num_instances(self) -> int:
        return self.num_instances

    def get_num_instance_slots(self) -> int:
        return len(self.instance_slots)

    def get_prompt_tokens_acc(self) -> int:
        return self.prompt_tokens_acc

    def get_response_tokens_acc(self) -> int:
        return self.response_tokens_acc

    def clamp_num_instances(self, num_instances: int) -> int:
        return max(1, min(int(num_instances), len(self.instance_slots)))

    def scale_instances_to(self, num_instances: int) -> None:
        num_instances = self.clamp_num_instances(num_instances)
        self.instances = self.instance_slots[:num_instances]
        self.num_instances = num_instances

    def get_instances_load_info(self) -> list[dict[str, Any]]:
        return [
            {
                "request_num": instance.get_instance_request_num(),
                "incoming_prefill_tokens": instance.get_instance_incoming_prefill_tokens(),
                "incoming_decode_tokens": instance.get_instance_incoming_decode_tokens(),
                "expected_token_usage": instance.get_instance_expected_token_usage(),
                "lookahead_max_tokens": instance.get_instance_lookahead_max_tokens(),
            }
            for instance in self.instances
        ]

    async def handle_request(self, request_id: int, request: Sequence[Any]) -> Dict[str, Any]:
        """Predict, route, and execute one request through PRISM."""

        prompt_len = int(request[1])
        true_response_len = int(request[2])
        if prompt_len < 1:
            raise ValueError(f"request {request_id} has invalid prompt length {prompt_len}")
        if true_response_len < 1:
            raise ValueError(f"request {request_id} has invalid response length {true_response_len}")
        prediction_start = time.time()
        predicted_output_len = self.perceptron.predict(prompt_len)
        if predicted_output_len + prompt_len > self.max_model_len:
            predicted_output_len = int(self.max_model_len) - prompt_len
        predicted_output_len = max(1, int(predicted_output_len))
        prediction_time = time.time() - prediction_start

        route_start = time.time()
        decision = self.router.choose_instance(self.instances, prompt_len, predicted_output_len)
        route_time = time.time() - route_start
        instance = decision.instance

        self.prompt_tokens_acc += prompt_len
        self.response_tokens_acc += true_response_len

        if self.dry_run:
            instance.request_num += 1
            instance.incoming_prefill_tokens += prompt_len
            instance.incoming_decode_tokens += predicted_output_len
            instance.expected_token_usage += prompt_len + predicted_output_len
            try:
                await asyncio.sleep(0)
                result = {
                    "request_id": request_id,
                    "instance_id": instance.get_instance_id(),
                    "request_finished": True,
                    "prompt_tokens": prompt_len,
                    "expected_tokens": predicted_output_len,
                    "generated_tokens": true_response_len,
                    "record_time": time.time(),
                    "latency": 0.0,
                    "error": "",
                    "dry_run": True,
                }
            finally:
                instance.request_num -= 1
                instance.incoming_prefill_tokens -= prompt_len
                instance.incoming_decode_tokens -= predicted_output_len
                instance.expected_token_usage -= prompt_len + predicted_output_len
        else:
            result = await instance.request_inference(request_id, request, predicted_output_len)

        if result.get("request_finished", False):
            observed_response_len = int(result.get("generated_tokens", true_response_len))
            if observed_response_len <= 0:
                observed_response_len = true_response_len
            self.perceptron.update(prompt_len, observed_response_len)

        result.update({
            "prism_predicted_output_len": predicted_output_len,
            "prism_prediction_time": prediction_time,
            "prism_routing_time": route_time,
            "prism_route_cost": decision.cost,
            "prism_route_prefill_latency": decision.prefill_latency,
            "prism_route_decode_latency": decision.decode_latency,
            "prism_route_heterogeneity": decision.heterogeneity,
            "prism_route_congestion": decision.congestion,
            "prism_runtime_notes": {
                "intra_instance_preemption": self.prism_config.implementation_notes["intra_instance_preemption"],
            },
        })
        return result


class SimulationInstance:
    """Serving-instance model used for deterministic PRISM dry runs."""

    def __init__(self, instance_id: int, instance_config: Dict[str, Any], scheduler_config: Dict[str, Any]):
        self.instance_id = instance_id
        self.instance_config = instance_config
        self.scheduler_config = scheduler_config
        self.request_num = 0
        self.incoming_prefill_tokens = 0
        self.incoming_decode_tokens = 0
        self.expected_token_usage = 0
        self.gpu_utilization = 0.0
        self.gpu_memory_utilization = 0.0
        self.lookahead_max_tokens = 0

    def get_instance_id(self):
        return self.instance_id

    def get_instance_request_num(self):
        return self.request_num

    def get_instance_incoming_prefill_tokens(self):
        return self.incoming_prefill_tokens

    def get_instance_incoming_decode_tokens(self):
        return self.incoming_decode_tokens

    def get_instance_expected_token_usage(self):
        return self.expected_token_usage

    def get_instance_lookahead_max_tokens(self):
        return self.lookahead_max_tokens
