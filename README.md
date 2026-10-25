# Transaction Fraud Risk Scorer

An explainable machine-learning mini-project that estimates transaction fraud probability from amount, geographic distance, transaction hour, and one-hour transaction velocity.

## Overview

Fraud screening systems help prioritize transactions for review. This educational project demonstrates a complete binary-classification workflow with feature standardization and logistic regression implemented using only Python's standard library.

## Problem statement

Given transaction behavior signals, estimate the probability that a transaction is fraudulent. The output is a triage signal, not a definitive fraud decision.

## Features

- Standardizes numeric transaction features.
- Trains logistic regression with gradient descent.
- Returns a probability and review recommendation.
- Includes labeled sample data, tests, and CLI.
- Uses no third-party runtime dependencies.

## Technology stack

- Python 3.10+
- Logistic regression and gradient descent
- Feature standardization
- CSV, `argparse`, and `math`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/transaction-fraud-risk-scorer.git
cd transaction-fraud-risk-scorer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party packages are required.

## Execution

```bash
python scorer.py --amount 45 --distance-km 4 --hour 14 --velocity-1h 2
```

Test a high-risk-looking transaction:

```bash
python scorer.py --amount 1200 --distance-km 1250 --hour 3 --velocity-1h 15
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 100.0%
Fraud probability: 0.2%
Likely legitimate
```

## Responsible use and limitations

The dataset is synthetic and deliberately small. Real fraud systems require privacy controls, temporal validation, calibrated probabilities, fairness monitoring, concept-drift detection, secure feature pipelines, and human review. Never decline a customer solely from this educational model.

## Future improvements

- Add time-based train/test evaluation and class-imbalance metrics.
- Calibrate probabilities and tune decision thresholds by review capacity.
- Add explainability showing feature contributions.
- Compare with tree ensembles and anomaly-detection models.
- Add drift and fairness monitoring dashboards.
