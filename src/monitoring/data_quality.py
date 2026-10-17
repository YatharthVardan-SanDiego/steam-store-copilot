"""Data-quality monitoring with SageMaker Model Monitor (batch setting).

The tag gate runs as batch transform, so instead of an endpoint schedule we run
the Model Monitor analyzer container directly against each incoming batch:

1. ``suggest_baseline``  - statistics.json + constraints.json from training data.
2. ``run_data_quality_check`` - same analyzer, compared to the baseline,
   writes constraint_violations.json for the batch.

``psi_report`` is a lightweight, dependency-free drift check (Population
Stability Index) used for the CloudWatch metric and unit tests.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

MONITOR_COLUMNS = ["price", "required_age", "desc_chars", "desc_words",
                   "windows", "mac", "linux", "n_tags_22"]


def monitor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tabular view monitored for drift (same columns as prepare.py's baseline)."""
    out = df.copy()
    if "n_tags_22" not in out:
        out["n_tags_22"] = out["tags_22"].map(lambda v: len(v.split("|")) if v else 0)
    for col in ("windows", "mac", "linux"):
        out[col] = out[col].astype(str).str.lower().isin(["true", "1"]).astype(int)
    return out[MONITOR_COLUMNS]


def suggest_baseline(role: str, baseline_csv_uri: str, output_uri: str, sagemaker_session):
    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat

    monitor = DefaultModelMonitor(
        role=role, instance_count=1, instance_type="ml.m5.xlarge", volume_size_in_gb=20,
        max_runtime_in_seconds=1800, base_job_name="tag-gate-baseline",
        sagemaker_session=sagemaker_session,
    )
    monitor.suggest_baseline(baseline_dataset=baseline_csv_uri,
                             dataset_format=DatasetFormat.csv(header=True),
                             output_s3_uri=output_uri, wait=True, logs=False)
    return monitor


def run_data_quality_check(role: str, dataset_uri: str, baseline_uri: str, output_uri: str,
                           sagemaker_session) -> str:
    """Run the Model Monitor analyzer on one batch; returns the report S3 prefix."""
    import sagemaker
    from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

    image = sagemaker.image_uris.retrieve("model-monitor", sagemaker_session.boto_region_name)
    processor = Processor(
        role=role, image_uri=image, instance_count=1, instance_type="ml.m5.xlarge",
        base_job_name="tag-gate-data-quality", sagemaker_session=sagemaker_session,
        env={
            "dataset_format": json.dumps({"csv": {"header": True}}),
            "dataset_source": "/opt/ml/processing/input/data",
            "output_path": "/opt/ml/processing/output",
            "publish_cloudwatch_metrics": "Disabled",
            "baseline_constraints": "/opt/ml/processing/baseline/constraints/constraints.json",
            "baseline_statistics": "/opt/ml/processing/baseline/stats/statistics.json",
        },
    )
    processor.run(
        inputs=[
            ProcessingInput(dataset_uri, "/opt/ml/processing/input/data"),
            ProcessingInput(f"{baseline_uri}/constraints.json",
                            "/opt/ml/processing/baseline/constraints"),
            ProcessingInput(f"{baseline_uri}/statistics.json",
                            "/opt/ml/processing/baseline/stats"),
        ],
        outputs=[ProcessingOutput("/opt/ml/processing/output", output_uri, output_name="report")],
        wait=True, logs=False,
    )
    return output_uri


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of ``actual`` vs ``expected`` (quantile bins)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant or binary feature: compare category shares
        cats = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == c).mean() for c in cats])
        a = np.array([(actual == c).mean() for c in cats])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(baseline: pd.DataFrame, batch: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """PSI per monitored column; PSI > 0.2 is the usual 'significant shift' rule."""
    rows = [{"feature": c, "psi": psi(baseline[c], batch[c])} for c in MONITOR_COLUMNS]
    report = pd.DataFrame(rows)
    report["drift"] = report["psi"] > threshold
    return report
