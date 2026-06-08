import unittest

from LLMServe.prism.routing import PrismRouter, pressure_score


class FakeInstance:
    def __init__(self, instance_id, request_num=0, prefill=0, gpu=0.0, mem=0.0):
        self.instance_id = instance_id
        self.request_num = request_num
        self.prefill = prefill
        self.gpu_utilization = gpu
        self.gpu_memory_utilization = mem

    def get_instance_id(self):
        return self.instance_id

    def get_instance_request_num(self):
        return self.request_num

    def get_instance_incoming_prefill_tokens(self):
        return self.prefill


class PrismRouterTest(unittest.TestCase):
    def test_choose_lower_cost_instance(self):
        router = PrismRouter(routing_prefill_alpha=1e-6, routing_decode_beta=1e-3, routing_congestion_weight=1.0)
        idle = FakeInstance(0, gpu=0.0)
        busy = FakeInstance(1, gpu=100.0)
        decision = router.choose_instance([busy, idle], prompt_len=100, predicted_output_len=100)
        self.assertEqual(decision.instance_id, 0)

    def test_heterogeneity_penalty_prefers_similar_prompt_queue(self):
        router = PrismRouter(
            routing_prefill_alpha=1e-3,
            routing_decode_beta=1e-6,
            routing_heterogeneity_weight=10.0,
        )
        similar = FakeInstance(0, request_num=2, prefill=200)
        different = FakeInstance(1, request_num=2, prefill=2000)
        decision = router.choose_instance([different, similar], prompt_len=100, predicted_output_len=100)
        self.assertEqual(decision.instance_id, 0)

    def test_pressure_score_increases_with_remaining_work(self):
        low = pressure_score(predicted_output_len=100, decoded_len=90, elapsed_seconds=10)
        high = pressure_score(predicted_output_len=100, decoded_len=10, elapsed_seconds=10)
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
