import unittest
from pathlib import Path

from scorer import LogisticFraudModel, StandardScaler, accuracy, load_csv, train

DATA = Path(__file__).parents[1] / "data" / "transactions.csv"


class ScorerTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(labels), 8)

    def test_scaler_centers_features(self):
        transformed = StandardScaler().fit([[1, 10], [3, 20]]).transform([[1, 10], [3, 20]])
        self.assertAlmostEqual(sum(row[0] for row in transformed), 0.0)

    def test_model_learns_training_examples(self):
        model, scaler, score = train(DATA)
        self.assertGreaterEqual(score, 0.95)
        low = model.probabilities(scaler.transform([[45, 4, 14, 2]]))[0]
        high = model.probabilities(scaler.transform([[1200, 1250, 3, 15]]))[0]
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_model_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            LogisticFraudModel().fit([], [])

    def test_accuracy(self):
        self.assertEqual(accuracy([0, 1], [0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
