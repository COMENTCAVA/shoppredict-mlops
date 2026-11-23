
# ShopPredict — Dataset Documentation

## 1. Data source

- Dataset: Online Shoppers Purchasing Intention
- Source: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- License: CC BY 4.0
- Download date: 2026-11-23

The dataset contains 12,330 e-commerce browsing sessions.

Each row represents one session. Revenue indicates whether the session resulted in a purchase.

## 2. Data cleaning

The original dataset contains 125 duplicate rows.

After removing duplicates, 12,205 unique sessions remain.

The following transformations are applied in ml/split.py:

- Remove duplicate rows.
- Convert Revenue to 0 or 1.
- Convert Weekend to 0 or 1.
- Split the data chronologically by month.
- Export deterministic CSV files.

The split is reproducible: repeated executions produce identical SHA-256 hashes.

## 3. Dataset versions

| Dataset | Months | Rows | Purchase rate |
|---|---|---:|---:|
| V1 | Feb, Mar, May, June, Jul, Aug | 6,520 | 11.21% |
| V2 | V1 + Sep, Oct, Nov | 10,499 | 16.12% |
| Evaluation | Dec | 1,706 | 12.66% |

V1 is used for initial model training.

V2 adds 3,979 sessions from September, October and November to simulate the arrival of new training data.

December is reserved for evaluation and must never be used for training or hyperparameter tuning.

The Month column is retained for traceability but excluded from model features because December is absent from the training data.

The model uses 16 input features, with Revenue as the binary target.

## 4. Data versioning with DVC

Data is tracked using DVC and stored in Azure Blob Storage.

- Storage account: wut2026
- Container: dvc
- DVC remote: myazure
- Remote URL: azure://dvc/shoppredict-mlops

### Version V1

- Git tag: data-v1
- Git commit: 5c9a78c
- DVC file: data/v1.dvc
- DVC MD5: 02fd83f80c363c458134f3ac1895b499.dir

### Version V2

- Git tag: data-v2
- Git commit: 2b083f5
- DVC file: data/v2.dvc
- DVC MD5: 733eda7bb552030ec3b710b3b298a68c.dir

### Evaluation dataset

- DVC file: data/eval.csv.dvc
- DVC MD5: cc979923875dc3fc2c16d11f7cb7cf58

## 5. Reproducibility

Generate the datasets:

    python -m ml.split

Download DVC-tracked datasets:

    dvc pull

Upload datasets to Azure:

    dvc push

Azure credentials must be configured locally and must never be committed to Git.

The original raw CSV must be downloaded separately from UCI.
