"""Explainable transaction fraud scoring with logistic regression."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

FEATURES = ("amount", "distance_km", "hour", "velocity_1h")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


class StandardScaler:
    def fit(self, rows: Sequence[Sequence[float]]) -> "StandardScaler":
        if not rows:
            raise ValueError("At least one row is required")
        self.means = [sum(row[i] for row in rows) / len(rows) for i in range(len(rows[0]))]
        self.scales = []
        for i, mean in enumerate(self.means):
            variance = sum((row[i] - mean) ** 2 for row in rows) / len(rows)
            self.scales.append(math.sqrt(variance) or 1.0)
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)] for row in rows]


class LogisticFraudModel:
    def __init__(self, learning_rate: float = 0.12, epochs: int = 1600) -> None:
        self.learning_rate, self.epochs = learning_rate, epochs

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> "LogisticFraudModel":
        if not rows or len(rows) != len(labels):
            raise ValueError("Rows and labels must be non-empty and equally sized")
        self.weights = [0.0] * (len(rows[0]) + 1)
        for _ in range(self.epochs):
            gradients = [0.0] * len(self.weights)
            for row, label in zip(rows, labels):
                probability = sigmoid(self.weights[0] + sum(weight * value for weight, value in zip(self.weights[1:], row)))
                error = probability - label
                gradients[0] += error
                for index, value in enumerate(row, 1):
                    gradients[index] += error * value
            for index, gradient in enumerate(gradients):
                self.weights[index] -= self.learning_rate * gradient / len(rows)
        return self

    def probabilities(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [sigmoid(self.weights[0] + sum(weight * value for weight, value in zip(self.weights[1:], row))) for row in rows]

    def predict(self, rows: Sequence[Sequence[float]], threshold: float = 0.5) -> list[int]:
        return [int(probability >= threshold) for probability in self.probabilities(rows)]


def load_csv(path: str | Path) -> tuple[list[list[float]], list[int]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return ([ [float(row[name]) for name in FEATURES] for row in rows], [int(row["is_fraud"]) for row in rows])


def accuracy(actual: Sequence[int], predicted: Sequence[int]) -> float:
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def train(path: str | Path) -> tuple[LogisticFraudModel, StandardScaler, float]:
    rows, labels = load_csv(path)
    scaler = StandardScaler().fit(rows)
    transformed = scaler.transform(rows)
    model = LogisticFraudModel().fit(transformed, labels)
    return model, scaler, accuracy(labels, model.predict(transformed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate transaction fraud probability.")
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--distance-km", type=float, required=True)
    parser.add_argument("--hour", type=float, required=True)
    parser.add_argument("--velocity-1h", type=float, required=True)
    args = parser.parse_args()
    model, scaler, score = train(args.data)
    probability = model.probabilities(scaler.transform([[args.amount, args.distance_km, args.hour, args.velocity_1h]]))[0]
    print(f"Training accuracy: {score:.1%}")
    print(f"Fraud probability: {probability:.1%}")
    print("Review required" if probability >= 0.5 else "Likely legitimate")
