"""CloudWatch dashboard: infrastructure (jobs) + model/business metrics."""

from __future__ import annotations

import json

from src.config import CLOUDWATCH_NAMESPACE, DASHBOARD_NAME


def _search_widget(title: str, namespace: str, metric: str, x: int, y: int, region: str) -> dict:
    expr = f"SEARCH('{{{namespace},Host}} MetricName=\"{metric}\"', 'Average', 60)"
    return {"type": "metric", "x": x, "y": y, "width": 8, "height": 6,
            "properties": {"title": title, "region": region, "view": "timeSeries",
                           "metrics": [[{"expression": expr, "id": "e1"}]], "stat": "Average"}}


def _custom_widget(title: str, metrics: list[str], x: int, y: int, region: str) -> dict:
    return {"type": "metric", "x": x, "y": y, "width": 8, "height": 6,
            "properties": {"title": title, "region": region, "view": "timeSeries",
                           "stat": "Maximum", "period": 300,
                           "metrics": [[CLOUDWATCH_NAMESPACE, m, "Model", "tag-gate"]
                                       for m in metrics]}}


def dashboard_body(region: str) -> dict:
    widgets = [
        {"type": "text", "x": 0, "y": 0, "width": 24, "height": 2,
         "properties": {"markdown": "# Steam Store Copilot - Tag QA Gate\n"
                                    "Infrastructure (SageMaker jobs) and model/business metrics."}},
        _search_widget("Training jobs - CPU %", "/aws/sagemaker/TrainingJobs",
                       "CPUUtilization", 0, 2, region),
        _search_widget("Training jobs - memory %", "/aws/sagemaker/TrainingJobs",
                       "MemoryUtilization", 8, 2, region),
        _search_widget("Processing jobs - CPU %", "/aws/sagemaker/ProcessingJobs",
                       "CPUUtilization", 16, 2, region),
        _search_widget("Batch transform - CPU %", "/aws/sagemaker/TransformJobs",
                       "CPUUtilization", 0, 8, region),
        _search_widget("Batch transform - memory %", "/aws/sagemaker/TransformJobs",
                       "MemoryUtilization", 8, 8, region),
        _custom_widget("Drifted features (PSI > 0.2)", ["drifted_features"], 16, 8, region),
        _custom_widget("Gate business metrics",
                       ["auto_approval_rate", "false_approval_rate", "human_review_rate"],
                       0, 14, region),
        _custom_widget("Batch macro-F1", ["macro_f1"], 8, 14, region),
        _custom_widget("Records scored", ["records"], 16, 14, region),
    ]
    return {"widgets": widgets}


def put_dashboard(cloudwatch_client, region: str) -> str:
    cloudwatch_client.put_dashboard(DashboardName=DASHBOARD_NAME,
                                    DashboardBody=json.dumps(dashboard_body(region)))
    return (f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
            f"#dashboards:name={DASHBOARD_NAME}")
