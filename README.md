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
- **Model Registry** — versioned tag and reception models
- **Training Pipeline** — automated SageMaker Pipeline for preprocessing, training, evaluation, and registration
- **Batch Transform** — batch inference (no real-time endpoint; drafts are reviewed in batches)
- **Model Monitor** — data-quality and model-quality monitoring with an automated retraining trigger
- **CI/CD** — pipeline demonstrable in both passing and failing states

## Data

Two public Steam datasets, joined on Steam application ID (73.0% coverage → 37,148 titles).

| File | Records | Source / License | Role |
|---|---|---|---|
| `recommendations.csv` | 41,154,794 | [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) (CC0) | Labels for reception model |
| `users.csv` | ~14,000,000 | same | Reviewer-history features |
| `recs/games.csv` | 50,872 | same | Game-level ratings and pricing |
| `games_metadata.json` | 50,872 | same | Secondary descriptions and tags |
| `store/games.csv` | 125,855 | [Steam Games Dataset — Fronkon Games](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) (MIT) | Descriptions, tags, metadata |

Raw data lives in **S3**, not in this repository.

### ⚠️ Known data issue: malformed header

The Fronkon `games.csv` header declares **39** fields, but every data row has **40** (a comma is missing between `Discount` and `DLC count`). A naive `pd.read_csv` silently shifts every column from index 7 onward — `Tags` ends up holding genre data and 452 tags collapse to 33.

**Always load this file through the shared loader** (`src/data/loaders.py`), which pins an explicit 40-column schema. Do not read it directly in notebooks.

### Preparation rules

- Keep games with a description ≥ 50 characters **and** a non-empty tag list (83,186 of 125,855 games, 66.1%)
- Drop predominantly non-Latin-script descriptions (~1.6%) — English-only for v1
- Strip residual BBCode (~0.4% of descriptions; no HTML present)
- Split reviews **chronologically**, holding out the most recent period as the test set

### Label spaces

- **Tags:** fixed at the 22 tags with ≥ 10,000 games each. The next tags (Funny: 9,841, Horror: 9,653) fall below the cutoff and are excluded.
- **Reviews:** `is_recommended` — 85.78% positive / 14.22% negative (5,850,396 minority-class examples)

## Modeling

| | Tag QA Gate | Reception Estimator |
|---|---|---|
| **Baseline** | TF-IDF + linear model | — |
| **Model** | Fine-tuned transformer encoder, 22-output sigmoid head | Gradient-boosted trees over text embeddings + tabular features |
| **Features** | Draft-time only: description text, length, price, platform flags, required age, developer/publisher | Draft-time + post-release features |
| **Primary metric** | Macro-F1 across 22 tags, per-tag thresholds | PR-AUC on minority class + calibration check |
| **Reported** | Per-tag precision and recall | — |

Where the gap to the baseline is narrow, the smaller model wins — retraining cost matters more than the last point of accuracy.

### Business metrics

- **Auto-approval rate** — share of drafts approved without human review (the workload saving)
- **False-approval rate** — the guardrail. We fix a ceiling on this and maximize auto-approval subject to it.

## Goals

- [ ] Complete, operable ML system on SageMaker (feature store, model registry, training pipeline, batch inference, CI/CD)
- [ ] Beat the TF-IDF baseline on macro-F1; publish per-tag precision and recall
- [ ] Hold false-approval rate at or below the agreed ceiling while maximizing auto-approval
- [ ] Drift detection with automated retraining and a demonstrated end-to-end run
- [ ] Fully reproducible: code in GitHub, data in S3, pinned schema loader

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
├── requirements.txt
├── .gitignore
├── docs/
│   └── M1_RFC_Steam_Store_Copilot.pdf   # project proposal
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_tag_model_baseline.ipynb
│   └── 04_reception_model.ipynb
├── src/
│   ├── data/
│   │   ├── loaders.py        # pinned 40-column schema loader — use this!
│   │   └── preprocess.py     # filtering, language filter, BBCode cleanup, splits
│   ├── features/             # feature store definitions (draft-time vs. post-release)
│   ├── models/
│   │   ├── tag_gate/         # multi-label tag classifier
│   │   └── reception/        # review reception model
│   ├── pipelines/            # SageMaker Pipeline definitions
│   └── monitoring/           # Model Monitor configs, drift + retraining trigger
├── tests/
└── .github/
    └── workflows/            # CI/CD
```


## Team

| Name | Role |
|---|---|
| Yatharth Vardan | IC |


