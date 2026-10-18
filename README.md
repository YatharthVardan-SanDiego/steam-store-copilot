# Steam Store Copilot

**An on-brand copy generation and QA system for indie game store pages.**

AAI-540 Machine Learning Operations — Final Project

---

## Overview

Northlight Interactive (a hypothetical digital distribution platform) hosts titles from ~4,000 small and independent studios. Most of these studios have no copywriter, so their store pages are inconsistent and often fail to signal what the game actually is — which hurts discovery on a storefront driven by search and tags.

Northlight wants to give studios an AI copywriting assistant, but its trust and safety team won't allow machine-drafted copy onto live pages without an automated check. **This project builds that check.**

| Component | What it does | ML task |
|---|---|---|
| **Tag QA Gate** (primary) | Reads a draft description and predicts the storefront tags a human curator would assign. Drafts whose predicted tags don't match the game's ground-truth tags are rejected and returned to the studio. | Supervised multi-label text classification (22 tags) |
| **Reception Estimator** (secondary) | Estimates how a title's audience is likely to receive it, so the content team can prioritize scarce human review. | Supervised binary classification over user reviews |

> The LLM that drafts the copy is a hosted, external dependency. The models trained, versioned, and monitored in this repo are the **QA gate** and the **reception estimator**.

## Architecture

```
 Studio inputs ──► Hosted LLM ──► Draft description
                                        │
                                        ▼
                         ┌──────────────────────────┐
                         │  Tag QA Gate (batch)     │  predicted tags vs. ground truth
                         └────────────┬─────────────┘
                              pass │        │ fail
                                   ▼        ▼
                     Reception Estimator   Returned to studio
                                   │
                                   ▼
                  Auto-approve / prioritize for human audit
```

Built on **Amazon SageMaker**:

- **Feature Store** — separate *draft-time* and *post-release* feature groups, so leakage-prone features (review counts, playtime, CCU, ownership) can never reach the tag model
- **Training Pipeline** — SageMaker Pipeline: data validation → baseline + candidate training → evaluation → quality gate → registration (or failure)
- **Model Registry** — versioned tag-gate and reception models with attached evaluation metrics and manual approval
- **Batch Transform** — batch inference with the latest *Approved* model (no real-time endpoint; drafts are reviewed in batches)
- **Model Monitor + CloudWatch** — data-quality baseline and checks, PSI drift, model/business metrics, dashboard, alarm, retraining trigger
- **CI/CD** — GitHub Actions (lint → unit tests → pipeline compile → deploy) and the pipeline's own pass/fail quality gate

## Data

Two public Steam datasets, joined on Steam application ID (77.5% of the 50,872 recommendation-dataset games → 39,436 titles).

| File | Records | Source / License | Role |
|---|---|---|---|
| `recs/recommendations.csv` | 41,154,794 | [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) (CC0) | Labels for reception model |
| `recs/users.csv` | 14,306,064 | same | Reviewer-history features |
| `recs/games.csv` | 50,872 | same | Game-level pricing and platforms |
| `recs/games_metadata.json` | 50,872 | same | Secondary descriptions and tags |
| `store/games.csv` | 125,855 | [Steam Games Dataset — Fronkon Games](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) (MIT) | Descriptions, tags, metadata |

Raw data lives in **S3**, not in this repository. Layout (bucket = the account's SageMaker default bucket):

```
s3://<bucket>/steam-store-copilot/
├── raw/{store,recs}/...            # the five files above (notebook 00)
├── feature-store/                  # offline store of both feature groups (notebook 02)
├── curated/tag_gate_snapshot/      # Athena snapshot of draft-time features = pipeline input
├── pipeline/<execution-id>/        # splits, Model Monitor baseline data, evaluation.json
├── models/{baseline,improved}/     # training-job artifacts
├── reception/                      # reception model data, artifacts, predictions
├── batch/<run-id>/                 # batch-transform inputs, outputs, quality.json
└── monitoring/                     # Model Monitor baseline + violation reports
```

### ⚠️ Known data issue: malformed header

The Fronkon `games.csv` header declares **39** fields, but every data row has **40** (a comma is missing between `Discount` and `DLC count`). A naive `pd.read_csv` silently shifts every column from index 7 onward — `Tags` ends up holding genre data.

**Always load this file through the shared loader** (`src/data/loaders.py`), which pins an explicit 40-column schema. `tests/test_loaders.py` guards this in CI.

### Preparation rules

- Keep games with a description ≥ 50 characters **and** a non-empty tag list (83,186 of 125,855 games, 66.1%)
- Drop predominantly non-Latin-script descriptions — English-only for v1 (→ **81,927** games)
- Strip residual BBCode (~0.6% of descriptions; no HTML present)
- Tag-gate splits: games released on/after **2025-01-01** are the *incoming drafts* stream (5,855) used for batch inference and drift monitoring; older games are split 80/10/10 by a hash of `app_id` (60,934 / 7,461 / 7,677)
- Reviews are split **chronologically**: train < 2022-01, validation 2022-H1, test ≥ 2022-07

### Label spaces

- **Tags:** fixed at the 22 tags with ≥ 10,000 games each. The next tags (Funny: 9,841, Horror: 9,653) fall below the cutoff and are excluded.
- **Reviews:** `is_recommended` — 85.78% positive / 14.22% negative (5,850,396 minority-class examples)

## Modeling

| | Tag QA Gate | Reception Estimator |
|---|---|---|
| **Baseline** | Word TF-IDF + one-vs-rest logistic regression, 0.5 cut-off | Prevalence (random ranking) |
| **Model** | Word TF-IDF + draft-time tabular features, class-balanced logistic regression, per-tag thresholds | SageMaker built-in XGBoost |
| **Features** | Draft-time only: description text, length, price, platform flags, required age | Reviewer history, playtime, price, platforms, days since release, draft-time game features (no reception aggregates) |
| **Primary metric** | Macro-F1 across 22 tags | PR-AUC on the negative class + calibration check |
| **Reported** | Per-tag precision and recall | PR curve, calibration curve, title-level audit priority |

Where the gap to the baseline is narrow, the smaller model wins — retraining cost matters more than the last point of accuracy. A fine-tuned transformer is future work.

### Business metrics

- **Auto-approval rate** — share of genuine drafts approved without human review (the workload saving)
- **False-approval rate** — share of *mismatched* drafts (a description paired with another game's tags) that are approved. This is the guardrail: the gate threshold maximizes auto-approval subject to a **5% ceiling**.

### Results (held-out test split)

| Metric | Baseline | Candidate |
|---|---|---|
| Tag macro-F1 (22 tags) | 0.481 | **0.608** |
| Tag micro-F1 | 0.601 | 0.658 |
| Auto-approval rate | — | 42.0% |
| False-approval rate (ceiling 5%) | — | 4.6% |
| Reception PR-AUC (prevalence 0.152) | 0.152 | **0.38** |

On the **incoming 2025 batch** (5,855 drafts never seen in training, 20% deliberately mismatched) the approved model
reaches macro-F1 0.635, auto-approves 44.8% of genuine drafts and falsely approves 3.9% of mismatched ones. Model
Monitor flags real drift on this batch (fewer community tags on new releases, price distribution shift), which
triggered an automatic `retrain-on-drift` pipeline run.

Most of the tag-gate lift comes from per-tag threshold tuning (baseline + tuned thresholds reaches 0.607) — calibration matters more than capacity here.

## Goals

- [x] Complete, operable ML system on SageMaker (feature store, model registry, training pipeline, batch inference, CI/CD)
- [x] Beat the TF-IDF baseline on macro-F1; publish per-tag precision and recall
- [x] Hold false-approval rate at or below the 5% ceiling while maximizing auto-approval
- [x] Drift detection with a retraining trigger and a demonstrated end-to-end run
- [x] Fully reproducible: code in GitHub, data in S3, pinned schema loader

## Non-Goals

- Training or fine-tuning the generative (copywriting) model
- Building a game recommender
- Real-time / low-latency serving
- Multilingual support (English only for v1)
- Replacing human review — the gate triages; a sampled share of approvals still get human audit
- Predicting sales, revenue, or unit volume

## Repository Structure

```
steam-store-copilot/
├── README.md
├── requirements.txt / requirements-dev.txt / pyproject.toml
├── .github/workflows/ci.yml          # lint → unit tests → pipeline compile → deploy pipeline
├── notebooks/                        # run in order in SageMaker Studio
│   ├── 00_setup_data_ingestion.ipynb         # raw files → S3
│   ├── 01_data_exploration.ipynb             # EDA, header bug, label space, reviews
│   ├── 02_feature_engineering_feature_store.ipynb  # feature groups, ingestion, Athena snapshot
│   ├── 03_tag_model_development.ipynb        # baseline vs candidate, per-tag P/R, gate operating point
│   ├── 04_reception_model.ipynb              # XGBoost training job, batch transform, registry
│   ├── 05_training_pipeline_cicd.ipynb       # pipeline success + failure, model approval
│   ├── 06_batch_inference.ipynb              # batch transform with the approved tag gate
│   └── 07_monitoring.ipynb                   # Model Monitor, PSI, CloudWatch dashboard, retraining
├── src/
│   ├── config.py                     # resource names + S3 layout
│   ├── data/                         # loaders.py (pinned schema), preprocess.py, s3io.py
│   ├── features/feature_store.py     # draft-time vs post-release feature groups
│   ├── models/
│   │   ├── tag_gate/                 # model_lib.py, train.py, inference.py (SageMaker entry points)
│   │   └── reception/features.py     # review-level features, chronological split
│   ├── pipelines/
│   │   ├── tag_gate_pipeline.py      # SageMaker Pipeline definition
│   │   ├── batch.py                  # draft batches, latest approved model package
│   │   └── scripts/                  # prepare.py, evaluate.py (processing steps)
│   └── monitoring/                   # data_quality.py, model_quality.py, dashboard.py, retrain.py
├── scripts/run_pipeline.py           # upsert + start the pipeline (used by CD)
├── tests/                            # unit tests + offline pipeline compile
└── docs/                             # design document, video script
```

## Running on SageMaker (AWS Academy Learner Lab)

1. Start the Learner Lab, open **SageMaker Studio**, and launch a JupyterLab space (`ml.t3.large` or `ml.m5.large`; 8 GB RAM is enough because heavy work runs as SageMaker jobs).
2. Clone this repo (Git → Clone Repository). No other setup is needed: each notebook's first cell pins the SageMaker SDK to v2.
3. Run the notebooks **in order**. Approximate wall-clock: 00 (5–10 min), 01 (5), 02 (25–35, mostly waiting for the offline store), 03 (5), 04 (15), 05 (35, two pipeline runs), 06 (10), 07 (20).
4. Save each notebook with its outputs and commit it — the charts are part of the deliverable.

All resources use the execution role (`LabRole`) and the default SageMaker bucket, so no credentials are configured by hand.

### Local development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

## Contributing

1. Branch from `main` for each unit of work: `git checkout -b feature/<short-description>`
2. Commit notebooks **with** their chart outputs — graphics that explain the data live in the `.ipynb` files
3. Never commit raw data, credentials, or model artifacts — those belong in S3
4. Merge through a pull request so CI runs and the history shows each change

## Team

- **Yatharth Vardan**
- **Sachin Kumar Shukla**

## License

MIT — compatible with both source datasets (CC0 and MIT).

Done!!!
