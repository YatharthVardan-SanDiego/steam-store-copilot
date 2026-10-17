# Video Script — ML System Operation Validation

**Steam Store Copilot · AAI-540 Final Project**
Target length: **10 minutes** (rubric top band: not over 10 min). Two presenters, about 5 minutes each.
Record in 1080p with Zoom or QuickTime screen recording and export as `.mp4`.

| # | Time | Presenter | On screen | Rubric item |
|---|---|---|---|---|
| 1 | 0:00–1:00 | Yatharth | Title slide → README overview | Business use case |
| 2 | 1:00–2:00 | Yatharth | README architecture diagram | Architecture |
| 3 | 2:00–3:15 | Yatharth | SageMaker → Feature Store, notebook 02 | Feature store and feature groups |
| 4 | 3:15–4:30 | Yatharth | Notebook 03 charts, Studio → Pipelines DAG (success) | Training, CI/CD success |
| 5 | 4:30–5:15 | Sachin | Pipelines DAG (failed run), GitHub Actions green and red | CI/CD failed state |
| 6 | 5:15–6:15 | Sachin | Studio → Models → Model Registry (both groups) | Model registry |
| 7 | 6:15–7:15 | Sachin | Notebook 06 output, S3 batch output | Batch inference outputs |
| 8 | 7:15–8:45 | Sachin | Notebook 07 reports, CloudWatch dashboard | Monitoring reports, infra dashboards |
| 9 | 8:45–10:00 | Yatharth + Sachin | Closing slide | Risks, challenges, future work |

---

## 1. Business use case (Yatharth, ~60 s)

> Hi, we are Yatharth Vardan and Sachin Kumar Shukla, and this is Steam Store Copilot.
> Northlight Interactive is a hypothetical storefront hosting around four thousand small studios. Most have no copywriter,
> so store pages are inconsistent and don't signal what the game is, which hurts discovery on a store driven by search
> and tags. Northlight wants an AI copywriting assistant, but trust and safety won't allow machine-drafted copy onto live
> pages without an automated check. **We built that check**: a Tag QA Gate that reads a draft and predicts the tags a
> curator would assign. If they don't match the game's real tags, the draft goes back to the studio. A second model, the
> Reception Estimator, flags titles likely to be received poorly so the content team can prioritize human audits.
> Success is measured by two business metrics: the **auto-approval rate**, which is the workload we save, and the
> **false-approval rate**, our guardrail, which we cap at **5 percent**.

## 2. Architecture (Yatharth, ~60 s)

Show the diagram in the README.

> Studio inputs go to a hosted LLM, which produces a draft. Drafts are scored in batches by the Tag QA Gate. Passing
> drafts are approved or prioritized by the Reception Estimator; failing drafts return to the studio. Everything runs on
> SageMaker: raw data in S3, two feature groups, a SageMaker Pipeline for training with a quality gate, the Model
> Registry, Batch Transform, and Model Monitor with CloudWatch. GitHub Actions runs CI on every push.

## 3. Feature Store (Yatharth, ~75 s)

Console: **SageMaker → Feature Store → Feature groups**. Open `steam-draft-features`, then `steam-post-release-features`.

> We use two feature groups keyed by Steam app id. The draft-time group holds only what a studio knows before launch:
> the cleaned description, length, price, age rating, platforms, and the label tags. Review counts, playtime,
> concurrent players and owners are post-release signals, so they live in a separate group. That split is our leakage
> guard: the tag model's training snapshot is queried from the draft-time offline store with Athena, and notebook 02
> asserts that no post-release column is present.

Show the online-store lookup cell (same game in both groups) and the Athena snapshot cell in notebook 02.

## 4. Training and CI/CD, success (Yatharth, ~75 s)

Notebook 03: show the macro-F1 bar chart and the auto-approval vs false-approval curve.

> The baseline is TF-IDF with logistic regression: macro-F1 0.48 on the held-out test set. Our candidate adds draft-time
> tabular features, class balancing and per-tag thresholds: 0.61. Honest finding: most of the lift comes from threshold
> tuning, which shows calibration matters more than capacity here. The gate threshold gives 42 percent auto-approval at
> a 4.6 percent false-approval rate, under our 5 percent ceiling.

Studio: **Pipelines → steam-tag-gate-pipeline → default-run** execution graph.

> The pipeline validates the data contract, trains the baseline and the candidate, evaluates both on held-out data, and
> the QualityGate registers the model only if macro-F1 clears the threshold and beats the baseline. This run passed and
> registered a new model version.

## 5. CI/CD, failed state (Sachin, ~45 s)

Studio: **quality-gate-fail-demo** execution graph (red `QualityGateFailed`).

> Here we ran the same pipeline with an unreachable bar of 0.95. Evaluation ran, the gate failed, and nothing was
> registered. A bad model can't reach production.

GitHub → **Actions**: show a green run (lint → unit tests → pipeline definition → deploy) and the red run from the PR that
reintroduced the CSV-header bug.

> On the code side, GitHub Actions lints, runs the unit tests, compiles the pipeline definition and, on main, deploys the
> pipeline. This red run is a pull request that broke the CSV loader; the test that guards the malformed header caught it
> before merge.

## 6. Model Registry (Sachin, ~60 s)

Studio: **Models → Registered models** → `steam-tag-gate` (versions, approval status, metrics), then `steam-reception-estimator`.

> Every pipeline run that passes the gate registers a version as Pending Manual Approval with its evaluation report
> attached. A human approves it, and only approved versions are used for inference. The Reception Estimator, an XGBoost
> model with PR-AUC 0.38 against a 15 percent base rate, is registered in its own group.

## 7. Batch inference outputs (Sachin, ~60 s)

Notebook 06: output table and the decisions chart; optionally the `.jsonl.out` file in S3.

> Batch Transform runs the latest approved model on incoming drafts: games released in 2025 that the model never saw.
> Twenty percent of drafts are deliberately paired with another game's tags, simulating copy written for the wrong game.
> Each output line has the predicted tags, an agreement score and a decision. On this batch the gate auto-approves 45
> percent of genuine drafts and lets through under 4 percent of mismatched ones.

## 8. Monitoring (Sachin, ~90 s)

Notebook 07: baseline statistics table, violations table, PSI chart. Then **CloudWatch → Dashboards → SteamStoreCopilot**.

> Model Monitor builds a baseline from the training features and checks every batch against it. Even the real 2025
> batch shows drift: new games carry fewer community tags, and prices shifted, so Model Monitor flags a violation. A
> simulated upstream failure, with truncated copy and a cents-versus-dollars pricing bug, triggers three violations and
> drift in five features. The CloudWatch dashboard combines infrastructure metrics (CPU and memory for training,
> processing and batch transform jobs) with our model and business metrics. When drift is detected, the retraining
> trigger starts the pipeline automatically, and the new run appears here as retrain-on-drift.

## 9. Risks, challenges and future work (both, ~75 s)

**Yatharth (challenges):**
> The hardest data issue was the malformed CSV header, which silently shifted columns and turned genres into tags; we
> pinned the schema and test it in CI. The second was memory: building the dataset on a 4-gigabyte Studio instance froze
> the kernel, so we rewrote it to stay under one gigabyte.

**Sachin (risks and future work):**
> Risks: the gate only checks tag consistency, not factual accuracy or tone, so a sampled share of approvals still gets
> human review. The label space is English-only and fixed at 22 tags, and the false-approval estimate relies on simulated
> mismatches. Next steps: a fine-tuned transformer if the macro-F1 gain justifies retraining cost, multilingual support,
> scheduled Model Monitor jobs with real batch data capture, and ground-truth feedback from curators.

> Thank you.

---

## Recording checklist

- [ ] Learner Lab started, credentials valid; Studio, CloudWatch and GitHub tabs open before recording
- [ ] Both pipeline executions visible (success and failed)
- [ ] Model Registry: at least one **Approved** tag-gate version
- [ ] CloudWatch dashboard shows data (run notebook 07 within the last few hours)
- [ ] GitHub Actions: one green run and one red run
- [ ] Final cut ≤ 10:00; export `.mp4`; submit this script as the transcript
