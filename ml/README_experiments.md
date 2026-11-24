
# ShopPredict — Experiments

## 1. Objective

The objective is to predict whether an e-commerce browsing session
will result in a purchase.

Two classification algorithms were tested:

- Logistic Regression
- Random Forest

Both models use a complete scikit-learn Pipeline with
OneHotEncoder, StandardScaler and balanced class weights.

The Month column is excluded from the model features.

## 2. Evaluation metrics

ROC-AUC is the primary metric because the dataset is imbalanced
and the application predicts purchase probabilities.

Accuracy alone could be misleading because most sessions do not
result in a purchase.

Additional metrics:

- F1-score: balance between precision and recall.
- Average Precision: evaluates the ranking of positive predictions.

All models are evaluated on the same 1,706 December sessions.
These observations are excluded from model fitting.

The evaluation set is also used for model selection and promotion,
so it is not a fully independent final test set.

The classification threshold is 0.5.

## 3. Experiment tracking

Experiments are tracked using MLflow on DagsHub.

- Tracking URI: https://dagshub.com/cone/shoppredict-mlops.mlflow
- Experiment: shoppredict
- Registered model: shoppredict
- Production alias: champion

Each training run records:

- Git commit SHA
- Dataset version
- DVC MD5 hash
- Model hyperparameters
- Evaluation metrics
- Confusion matrix
- Complete trained scikit-learn Pipeline

## 4. Baseline experiments on V1

Training dataset: 6,520 sessions.

Evaluation dataset: 1,706 sessions.

| Model | ROC-AUC | F1 | Average Precision |
|---|---:|---:|---:|
| Logistic Regression | 0.8851 | 0.6181 | 0.6140 |
| Random Forest | 0.9118 | 0.6432 | 0.7265 |

### Logistic Regression

- Dataset: v1
- MLflow model version: 1
- Run ID: 089fb9b530aa40df83792c3c4789b451
- class_weight: balanced
- max_iter: 1000

### Random Forest

- Dataset: v1
- MLflow model version: 2
- Run ID: 03a67770f5aa415f8fb92b93ceb42d0a
- class_weight: balanced
- random_state: 42

Random Forest was selected as the baseline because it achieved
better results across all three evaluation metrics.

A repeated Random Forest experiment produced identical metrics.

- MLflow model version: 3
- Run ID: 8fd2e7cae95548109bacdfb370585f65

## 5. Quality gates

The minimum ROC-AUC threshold was frozen on 2026-11-24,
before evaluating V2.

Git commit: bd39d64

The policy is implemented in ml/policy.py.

| Parameter | Value | Description |
|---|---:|---|
| FLOOR | 0.85 | Minimum acceptable ROC-AUC |
| MIN_GAIN | 0.01 | Minimum improvement for promotion |
| MAX_DROP | 0.03 | Performance drop triggering retraining |

### Retraining rules

Retraining is triggered when at least one condition is met:

- A new validated dataset version is available.
- Champion ROC-AUC is below FLOOR.
- Champion ROC-AUC has dropped by at least MAX_DROP
  compared with its approved score.

### Promotion rules

A candidate can be promoted only if:

- Its ROC-AUC is at least FLOOR.
- Its ROC-AUC exceeds the current champion's score
  by at least MIN_GAIN.

For the initial bootstrap, only the FLOOR condition is required.

## 6. Continuous Training results

### V1 — Bootstrap

Command:

    python -m ml.pipeline --version v1

- Algorithm: Random Forest
- Training sessions: 6,520
- MLflow model version: 4
- Run ID: bdd4f04243ff4ad9b982ba9e16dfea6c
- ROC-AUC: 0.9118133233905046
- Approved: true

Decision:

    {"decision": "bootstrap", "champion_version": "4", "reason": "Initial model meets quality floor"}

The model exceeded the minimum quality threshold and was
assigned the champion alias.

### V2 — Retraining

Command:

    python -m ml.pipeline --version v2

V2 adds 3,979 training sessions from September,
October and November.

- Algorithm: Random Forest
- Training sessions: 10,499
- MLflow model version: 5
- Run ID: 7454de1275de419584fbe7d80ce6f4ae
- ROC-AUC: 0.9036835073328362
- F1-score: 0.6313799621928167
- Average Precision: 0.706997120979372
- Approved: false

| Metric | Champion V1 | Candidate V2 |
|---|---:|---:|
| ROC-AUC | 0.9118 | 0.9037 |
| F1-score | 0.6432 | 0.6314 |
| Average Precision | 0.7265 | 0.7070 |

The candidate exceeded FLOOR but did not achieve
the required improvement of 0.01 over the champion.

Decision:

    {"decision": "retained", "champion_version": "4", "reason": "Candidate improvement is insufficient"}

The champion alias remained assigned to model version 4.

## 7. Rollback

The rollback command restores an approved model version.

    python -m ml.rollback --to-version 4

Only versions with the registry tag approved=true are eligible.

An attempt to roll back to version 5 was rejected:

    Rollback error: Model version 5 is not approved

After a successful rollback, the application must be restarted
or redeployed to load the selected champion.

## 8. Final champion

The selected champion at the end of these experiments is:

- Model: shoppredict
- MLflow version: 4
- Alias: champion
- Algorithm: Random Forest
- Dataset: v1
- ROC-AUC: 0.9118

The V2 experiment demonstrates that additional training data
does not necessarily improve predictive performance.

The promotion policy protects production by retaining the
existing champion when a candidate fails the quality gates.
