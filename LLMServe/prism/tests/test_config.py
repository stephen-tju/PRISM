import json
import os
import tempfile
import unittest

from LLMServe.prism.config import PrismConfig, _normalize_config_keys, load_prism_config


class PrismConfigTest(unittest.TestCase):
    def test_legacy_keys_normalize_to_research_names(self):
        normalized = _normalize_config_keys({
            "history_size": 32,
            "default_output_len": 128,
            "memory_capacity_tokens": 4096,
        })
        self.assertEqual(normalized["response_history_size"], 32)
        self.assertEqual(normalized["fallback_response_len"], 128)
        self.assertEqual(normalized["peak_memory_capacity_tokens"], 4096)
        self.assertNotIn("history_size", normalized)

    def test_invalid_threshold_order_fails_loudly(self):
        with self.assertRaises(ValueError):
            PrismConfig(scale_down_memory_threshold=0.9, scale_up_memory_threshold=0.8)

    def test_unknown_config_key_reports_value_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"unknown_prism_key": 1}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_prism_config(path)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
