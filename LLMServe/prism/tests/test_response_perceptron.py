import unittest

from LLMServe.prism.response_perceptron import ResponseLengthPerceptron


class ResponseLengthPerceptronTest(unittest.TestCase):
    def test_empty_history_uses_default(self):
        perceptron = ResponseLengthPerceptron(fallback_response_len=123)
        self.assertEqual(perceptron.predict(10), 123)

    def test_expectation_is_deterministic(self):
        perceptron = ResponseLengthPerceptron(
            response_history_size=4,
            response_recency_decay=1.0,
            prompt_similarity_sigma=1000.0,
            response_length_prior=1.0,
            response_prediction_mode="expectation",
        )
        perceptron.update(100, 100)
        perceptron.update(100, 300)
        self.assertEqual(perceptron.predict(100), 200)

    def test_prompt_similarity_affects_distribution(self):
        perceptron = ResponseLengthPerceptron(
            response_history_size=4,
            response_recency_decay=1.0,
            prompt_similarity_sigma=10.0,
            response_length_prior=0.1,
        )
        perceptron.update(10, 50)
        perceptron.update(1000, 500)
        self.assertLess(perceptron.predict(10), perceptron.predict(1000))


if __name__ == "__main__":
    unittest.main()
