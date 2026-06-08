import unittest

from LLMServe.prism.scaling import PeakWorkloadAnticipator


class FakeInstance:
    def __init__(self, instance_id, prefill=0, decode=0, expected=0, lookahead=0):
        self.instance_id = instance_id
        self.prefill = prefill
        self.decode = decode
        self.expected = expected
        self.lookahead = lookahead

    def get_instance_id(self):
        return self.instance_id

    def get_instance_incoming_prefill_tokens(self):
        return self.prefill

    def get_instance_incoming_decode_tokens(self):
        return self.decode

    def get_instance_expected_token_usage(self):
        return self.expected

    def get_instance_lookahead_max_tokens(self):
        return self.lookahead


class PeakWorkloadAnticipatorTest(unittest.TestCase):
    def test_estimate_uses_max_observable_pressure(self):
        anticipator = PeakWorkloadAnticipator(peak_memory_capacity_tokens=1000)
        estimate = anticipator.estimate_instance(
            FakeInstance(0, prefill=100, decode=200, expected=250, lookahead=900)
        )
        self.assertEqual(estimate.peak_tokens, 900)
        self.assertAlmostEqual(estimate.utilization, 0.9)
        self.assertTrue(estimate.should_scale_up)

    def test_desired_instances_scales_down_when_all_instances_low(self):
        anticipator = PeakWorkloadAnticipator(
            peak_memory_capacity_tokens=1000,
            scale_down_memory_threshold=0.2,
            scaling_violation_tolerance=0.2,
        )
        desired = anticipator.desired_instances(
            [FakeInstance(0, expected=10), FakeInstance(1, expected=20)],
            current_num_instances=2,
        )
        self.assertEqual(desired, 1)

    def test_desired_instances_scales_up_on_high_ratio(self):
        anticipator = PeakWorkloadAnticipator(
            peak_memory_capacity_tokens=1000,
            scale_up_memory_threshold=0.8,
            scaling_violation_tolerance=0.2,
        )
        desired = anticipator.desired_instances(
            [FakeInstance(0, expected=900), FakeInstance(1, expected=100)],
            current_num_instances=2,
        )
        self.assertEqual(desired, 3)

    def test_invalid_thresholds_fail_loudly(self):
        with self.assertRaises(ValueError):
            PeakWorkloadAnticipator(
                peak_memory_capacity_tokens=1000,
                scale_down_memory_threshold=0.9,
                scale_up_memory_threshold=0.8,
            )


if __name__ == "__main__":
    unittest.main()
