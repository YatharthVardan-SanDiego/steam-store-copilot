# ML System Design Document — Steam Store Copilot

**AAI-540 Machine Learning Operations · Final Project**
**Team:** Yatharth Vardan, Sachin Kumar Shukla
**Repository:** https://github.com/YatharthVardan-SanDiego/steam-store-copilot

> Paste each section into the matching heading of the AAI-540 Design Document Template. Numbers come from the executed
> notebooks in the repository (test split unless stated otherwise).

---

## 1. Problem Statement

Northlight Interactive, a hypothetical digital storefront, hosts titles from about 4,000 small and independent studios.
Most of these studios have no copywriter, so store-page descriptions vary widely in quality and often fail to signal what
the game actually is: its genre, perspective, art style and mode of play. On a storefront where discovery is driven by
search and by curator-assigned tags, a description that does not match the game's tags hurts both the studio, which
loses visibility, and the platform, which loses conversion. Northlight wants to offer studios an AI copywriting assistant
built on a hosted large language model, but its trust and safety team will not allow machine-drafted copy onto live
pages without an automated, auditable check.

This project builds that check as a machine learning system. The primary component, the **Tag QA Gate**, is a
supervised multi-label text classifier that reads a draft description and predicts which of 22 storefront tags a human
curator would assign. A draft whose predicted tags do not agree with the game's ground-truth tags is rejected and
returned to the studio; drafts that agree are approved automatically. A secondary component, the **Reception
Estimator**, is a supervised binary classifier over user reviews that estimates how likely a title is to be received
negatively, so the content team can prioritize its limited human-review capacity. The generative model itself is an
external dependency and is out of scope; the models trained, versioned, deployed and monitored here are the gate and
the estimator.

## 2. Impact Measurement

The system is judged on two business metrics that map directly onto the goal of letting more machine-drafted copy go
live without increasing risk. The **auto-approval rate** is the share of genuine drafts approved without human review;
it measures the workload saved. The **false-approval rate** is the share of *mismatched* drafts (a description paired
with a different game's tags, simulating copy written for the wrong game) that the gate approves; it is the safety
guardrail. We fix a ceiling of **5%** on false approvals and choose the gate threshold that maximizes auto-approval
subject to that ceiling. On the held-out test set the gate auto-approves **42.0%** of genuine drafts at a **4.6%**
false-approval rate, so roughly two in five drafts need no manual review while the guardrail holds.

Model-level metrics support these business metrics. For the gate we report **macro-F1 across the 22 tags**, so rare
tags count as much as common ones, plus per-tag precision and recall. The candidate model reaches **0.608** macro-F1
against **0.481** for the TF-IDF baseline (+0.127). For the Reception Estimator we report **PR-AUC on the negative
class**, which is appropriate for a 15% minority class, plus a calibration check: PR-AUC is **0.38** against a
random-ranking baseline of **0.152** (the prevalence). In production these metrics are recomputed on every batch and
published to CloudWatch; a drop of more than 0.05 macro-F1, a false-approval rate above the ceiling, or input drift
triggers retraining.

## 3. Security Checklist, Risks, Bias and Ethics

| Question | Answer |
|---|---|
| Does the project use PHI (health data)? | No. |
| Does it use PII? | No direct PII. The reviews dataset contains anonymized numeric `user_id`s with no names, emails or free text. Only aggregate reviewer history (games owned, reviews written) is used as features. |
| Does it use user behavior data? | Yes: review recommendations and hours played, anonymized and public (CC0). |
| Does it use credit card or payment data? | No. Prices are public list prices. |
| Where is data stored? Is it encrypted? | Raw and derived data are in the account's SageMaker default S3 bucket (SSE-S3 encryption at rest by default); the Feature Store offline store is in the same bucket; the online store is encrypted by AWS. No data is committed to GitHub (`.gitignore` blocks CSV/JSON data and credentials). |
| Access control | All jobs run under the lab execution role (`LabRole`); no long-lived keys are stored in code. CI deploy credentials, when used, are GitHub encrypted secrets. |
| Licensing | Source datasets are CC0 and MIT; the repository is MIT. |

**Bias.** The label space is fixed at the 22 most common tags (≥ 10,000 games each), so niche genres such as Horror
or Visual Novel are not represented and the gate cannot vouch for them; those drafts are judged only on the common
tags they carry. The training data is English-only (non-Latin descriptions were filtered out), so non-English studios
would be routed to human review rather than served by the gate. Tags are crowd-assigned on Steam and reflect community
conventions, which may encode popularity bias. Class-balanced training and per-tag thresholds reduce, but do not
remove, the gap between frequent and rare tags. Per-tag recall is published so these gaps are visible.

**Ethics and misuse risks.** The gate checks tag *consistency*, not factual accuracy, tone or policy compliance, so it
must not be presented as a full content-moderation system; a sampled share of approved drafts still receives human
audit. The Reception Estimator must not be used to penalize studios or to suppress games; it only orders the human
review queue. The false-approval estimate relies on simulated mismatches, which may be easier to catch than subtle
real-world errors, so the ceiling should be re-validated with curator feedback once the system is live.

## 4. Solution Overview

### 4.1 Data Sources

The system uses two public Steam datasets joined on the Steam application id. The **Steam Games Dataset** by Fronkon
Games (MIT, 125,855 games) provides store descriptions, curator tags, genres, price, platforms, age rating and
post-release statistics. The **Game Recommendations on Steam** dataset (CC0) provides 41,154,794 user reviews with a
binary `is_recommended` label and hours played, 14,306,064 anonymized users with their product and review counts, and
50,872 games with prices and platforms. 77.5% of the recommendation dataset's games (39,436) match a store-dataset
game, and 94.7% of reviews belong to a game with draft-time features.

The datasets satisfy the course data requirements: five files, four with far more than 10,000 records; 22 tag classes
with at least 10,000 labeled games each; and 5,850,396 negative and 35,304,398 positive reviews. All raw files are
stored in Amazon S3 under `s3://<sagemaker-default-bucket>/steam-store-copilot/raw/`, uploaded by notebook 00, and the
S3 layout is documented in the README.

### 4.2 Data Engineering

Exploration (notebook 01) found a defect that would have silently corrupted the primary label: the Fronkon
`games.csv` header declares 39 fields while every row has 40, because a comma is missing between `Discount` and
`DLC count`. A naive `pandas.read_csv` shifts every column from index 7 onward, and the `Tags` column ends up holding
genres. We wrote a shared loader (`src/data/loaders.py`) that skips the broken header and pins an explicit 40-column
schema, and a unit test reproduces the malformed header so CI catches any regression.

Cleaning rules (`src/data/preprocess.py`) keep games with a description of at least 50 characters and at least one tag
(83,186 of 125,855), drop predominantly non-Latin descriptions to keep v1 English-only (81,927 remain), and strip
residual BBCode (0.55% of descriptions). To avoid temporal leakage and to create a realistic production stream, games
released on or after 2025-01-01 (5,855) are held out as **incoming drafts** that are never trained on; older games are
split 80/10/10 into train, validation and test (60,934 / 7,461 / 7,677) by a hash of the app id, which is
deterministic and reproducible. Reviews for the Reception Estimator are split chronologically (train before 2022,
validation 2022-H1, test from 2022-07). The 41M-row review file is always streamed from S3 in chunks, and the dataset
build was rewritten to peak below 1 GB of RAM after it exhausted a 4 GB Studio instance.

### 4.3 Feature Engineering

Features are stored in **SageMaker Feature Store** (notebook 02) in two feature groups keyed by `app_id`, both with
online and offline stores. `steam-draft-features` contains only what a studio knows when it submits a draft: the
cleaned description, character and word counts, price, required age, platform flags, developer, publisher, release
date and the 22-tag label. `steam-post-release-features` holds review counts, positive ratio, peak concurrent users,
playtime and estimated owners. This separation is the main leakage guard: those post-release signals are strongly
correlated with the tags of established games but are unavailable at draft time.

The training snapshot for the pipeline is produced by an Athena query over the draft-time offline store that keeps the
latest record per game. The notebook asserts that no post-release column is present, and the pipeline's first step
re-validates the data contract (required columns, minimum rows, minimum positives per tag) before training. Model
features are a word-level TF-IDF (unigrams and bigrams, 200,000 terms, sublinear term frequency, English stop words)
plus log-scaled and standardized numeric features and platform flags. For the Reception Estimator, review-level
features combine reviewer history (products owned, reviews written), hours played, price, discount, platforms, days
since release and the game's draft-time tags; game-level aggregates that encode reception itself were excluded.

### 4.4 Model Training & Evaluation

The Tag QA Gate baseline is word TF-IDF with one-vs-rest logistic regression at a 0.5 cut-off (test macro-F1 0.481).
The candidate adds the tabular features, class-balanced logistic regression, and a per-tag decision threshold tuned on
validation to maximize each tag's F1 (test macro-F1 0.608, micro-F1 0.658). An ablation shows the baseline with tuned
thresholds reaches 0.607, so the lift comes mainly from calibration rather than model capacity. Character n-grams were
tried and dropped because they were about 20 times slower to train with no gain. The model trains in about 30 seconds,
which keeps retraining cheap, a design goal. The gate threshold (0.67 set-F1 agreement) is tuned on validation to
maximize auto-approval with false approvals at or below 5%.

Training is automated as a **SageMaker Pipeline** (`src/pipelines/tag_gate_pipeline.py`): PrepareData (validation and
splits) → TrainBaseline → TrainCandidate (SKLearn training jobs) → Evaluate (held-out test, writes
`evaluation.json`) → QualityGate. The gate registers the candidate only if test macro-F1 is at least the
`MacroF1Threshold` parameter **and** it beats the baseline; otherwise a FailStep stops the run. The Reception Estimator
is trained with the SageMaker built-in XGBoost algorithm on a 5% sample of reviews (about 2M rows), with
`scale_pos_weight` for imbalance and early stopping on validation PR-AUC. It reaches PR-AUC 0.38 (prevalence 0.152)
and ROC-AUC 0.75 on the most recent period. The calibration curve shows that class weighting inflates probabilities,
so scores are used for ranking titles, not as calibrated rates.

### 4.5 Model Deployment

Both models are versioned in the **SageMaker Model Registry** with their evaluation reports attached as model metrics:
`steam-tag-gate` (registered by the pipeline as PendingManualApproval) and `steam-reception-estimator`. Promotion is an
explicit human approval in the registry, which gives trust and safety a sign-off point and an audit trail.

Because drafts are reviewed in batches and no user is waiting on a response, deployment uses **SageMaker Batch
Transform** rather than a real-time endpoint, which removes always-on endpoint cost. Notebook 06 loads the latest
approved model package and scores the incoming-drafts stream (5,855 games released in 2025, 20% deliberately paired
with another game's tags) as JSON Lines: macro-F1 0.635, 44.8% of genuine drafts auto-approved, 3.9% of mismatched
drafts falsely approved, and 0.3% routed to human review. The custom inference handler (`src/models/tag_gate/inference.py`) returns
predicted tags, the agreement score, the top probabilities and a decision of approve, reject or human_review (the last
for games with none of the 22 tags). The Reception Estimator is deployed the same way over the most recent review
period, and its output ranks titles for human audit.

### 4.6 Model Monitoring

Monitoring (notebook 07, `src/monitoring/`) covers data quality, model quality and infrastructure. **SageMaker Model
Monitor** builds a baseline (statistics and constraints) from the training split's tabular features, and the Model
Monitor analyzer container then checks each batch against it and writes `constraint_violations.json`. We also compute a
Population Stability Index per feature (drift if PSI > 0.2). On the real 2025 batch, Model Monitor reported one
violation (price distribution drift, distance 0.10) and PSI flagged `n_tags_22` (0.29): newly released games carry
fewer community tags, because tags accumulate after release. This is genuine drift between training data and the
production stream, and it is the kind of shift the retraining policy exists for. A simulated upstream failure
(truncated LLM copy, a cents-versus-dollars price bug and invalid age ratings) produced three violations and PSI
values up to 7.3 across five features, which confirms that the checks catch a broken feed.

Model and business quality are computed from the batch output (macro-F1 on genuine drafts, auto-approval,
false-approval and human-review rates) and published as custom **CloudWatch** metrics. A CloudWatch dashboard
combines them with infrastructure metrics (CPU and memory of training, processing and transform jobs). A retraining
policy (`retrain.should_retrain`) fires on feature drift, data-quality violations, a macro-F1 drop above 0.05, or a
false-approval rate above the ceiling, and starts the training pipeline; in our run it started a `retrain-on-drift`
pipeline execution automatically. The same trigger is wired as a CloudWatch alarm (`steam-tag-gate-drift`) with an
EventBridge rule that starts the pipeline when the alarm fires. On the 2025 batch itself, model quality held up
(macro-F1 0.635, auto-approval 44.8%, false-approval 3.9%), so the drift is a leading indicator rather than an outage.

### 4.7 CI/CD

Continuous integration runs in **GitHub Actions** (`.github/workflows/ci.yml`) on every push and pull request as a
four-stage DAG: lint (ruff, including the notebooks) → unit tests (19 tests: the malformed-header loader, cleaning
rules, split determinism, gate logic, inference round trip, PSI drift and the retraining policy) → offline compilation
of the SageMaker Pipeline definition (no AWS calls) → deploy. On `main`, the deploy job upserts and starts the
SageMaker Pipeline when AWS credentials are configured as repository secrets, and skips cleanly otherwise because AWS
Academy credentials expire each session.

Continuous delivery of models is handled by the pipeline itself. Its QualityGate is the model-side CI check: we
demonstrate a successful run that registers a new version and a failed run (`MacroF1Threshold=0.95`) that evaluates
the candidate but blocks registration. On the code side, the Actions history shows a passing run and a failing run on a
pull request that reintroduced the CSV-header bug, caught by the loader test before merge. Together with manual
approval in the Model Registry, this gives three gates between a code change and production inference: tests, the
quality gate and human approval.

## 5. Future Improvements and Risks

- **Model:** fine-tune a small transformer encoder and adopt it only if the macro-F1 gain justifies the training cost;
  expand the label space beyond 22 tags as data grows.
- **Scope:** multilingual support; checks for tone and factual consistency, not just tags.
- **Monitoring:** scheduled Model Monitor jobs with batch data capture, and curator feedback as ground truth instead of
  simulated mismatches.
- **Scalability:** Feature Store ingestion was the slowest step (about 82k records per group); moving ingestion into
  a Processing job would parallelize it inside AWS.
- **Technical risk:** AWS Academy credentials expire every session, so fully automated CD needs a permanent IAM role
  in a production account.
