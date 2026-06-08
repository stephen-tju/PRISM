from __future__ import annotations

from collections import deque
import math
import random
from typing import Deque, Iterable, Tuple


class ResponseLengthPerceptron:
    """Online response-length estimator used by PRISM.

    The estimator maintains a sliding window of completed requests and builds a
    prompt-conditioned response-length distribution. Recency decay favors fresh
    observations, a Gaussian prompt kernel captures length similarity, and a
    Dirichlet prior keeps rare response lengths represented.
    """

    def __init__(
        self,
        response_history_size: int = 128,
        response_recency_decay: float = 0.96,
        prompt_similarity_sigma: float = 256.0,
        response_length_prior: float = 1.0,
        fallback_response_len: int = 256,
        response_prediction_mode: str = "expectation",
        random_seed: int = 7,
    ):
        if response_history_size <= 0:
            raise ValueError("response_history_size must be positive")
        if not 0 < response_recency_decay <= 1:
            raise ValueError("response_recency_decay must be in (0, 1]")
        if prompt_similarity_sigma <= 0:
            raise ValueError("prompt_similarity_sigma must be positive")
        if response_length_prior <= 0:
            raise ValueError("response_length_prior must be positive")
        if response_prediction_mode not in {"expectation", "sample"}:
            raise ValueError("response_prediction_mode must be 'expectation' or 'sample'")

        self.response_history_size = response_history_size
        self.response_recency_decay = response_recency_decay
        self.prompt_similarity_sigma = prompt_similarity_sigma
        self.response_length_prior = response_length_prior
        self.fallback_response_len = int(fallback_response_len)
        self.response_prediction_mode = response_prediction_mode
        self._history: Deque[Tuple[int, int]] = deque(maxlen=response_history_size)
        self._rng = random.Random(random_seed)

    def __len__(self) -> int:
        return len(self._history)

    def update(self, prompt_len: int, response_len: int) -> None:
        """Record an observed request after completion."""

        prompt_len = max(1, int(prompt_len))
        response_len = max(1, int(response_len))
        self._history.append((prompt_len, response_len))

    def update_many(self, rows: Iterable[Tuple[int, int]]) -> None:
        """Warm the estimator from historical ``(prompt_len, response_len)`` rows."""

        for prompt_len, response_len in rows:
            self.update(prompt_len, response_len)

    def distribution(self, prompt_len: int) -> dict[int, float]:
        """Return the smoothed conditional distribution for a prompt length."""

        if not self._history:
            return {self.fallback_response_len: 1.0}

        prompt_len = max(1, int(prompt_len))
        values = sorted({response_len for _, response_len in self._history})
        weights = {value: self.response_length_prior for value in values}

        total_history = len(self._history)
        for index, (hist_prompt_len, hist_response_len) in enumerate(self._history):
            recency_power = total_history - index - 1
            recency_weight = self.response_recency_decay ** recency_power
            distance = prompt_len - hist_prompt_len
            similarity_weight = math.exp(
                -(distance * distance) / (2 * self.prompt_similarity_sigma * self.prompt_similarity_sigma)
            )
            weights[hist_response_len] += recency_weight * similarity_weight

        total = sum(weights.values())
        return {value: weight / total for value, weight in weights.items()}

    def predict(self, prompt_len: int) -> int:
        """Predict a response length for an incoming request."""

        dist = self.distribution(prompt_len)
        if self.response_prediction_mode == "sample":
            threshold = self._rng.random()
            running = 0.0
            for response_len, probability in sorted(dist.items()):
                running += probability
                if running >= threshold:
                    return int(response_len)
            return int(max(dist))

        expected = sum(response_len * probability for response_len, probability in dist.items())
        return max(1, int(round(expected)))
