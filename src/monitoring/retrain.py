"""Retraining trigger: decide from monitoring results, then start the pipeline.

Two wiring options are provided:
* ``trigger_retraining`` - called from the monitoring job/notebook.
* ``create_alarm_and_rule`` - CloudWatch alarm on the drift metric plus an
  EventBridge rule that starts the SageMaker Pipeline when the alarm fires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.config import CLOUDWATCH_NAMESPACE, PIPELINE_NAME


@dataclass
class RetrainDecision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)


def should_retrain(drifted_features: list[str], dq_violations: int, batch_metrics: dict,
                   reference_macro_f1: float, max_f1_drop: float = 0.05,
                   max_false_approval: float = 0.05) -> RetrainDecision:
    reasons = []
    if drifted_features:
        reasons.append(f"feature drift (PSI>0.2): {', '.join(drifted_features)}")
    if dq_violations:
        reasons.append(f"{dq_violations} Model Monitor data-quality violations")
    drop = reference_macro_f1 - batch_metrics.get("macro_f1", reference_macro_f1)
    if drop > max_f1_drop:
        reasons.append(f"macro-F1 dropped {drop:.3f} (> {max_f1_drop})")
    if batch_metrics.get("false_approval_rate", 0.0) > max_false_approval:
        reasons.append(f"false-approval {batch_metrics['false_approval_rate']:.3f} "
                       f"above ceiling {max_false_approval}")
    return RetrainDecision(bool(reasons), reasons)


def trigger_retraining(sagemaker_client, decision: RetrainDecision,
                       pipeline_name: str = PIPELINE_NAME) -> str | None:
    if not decision.retrain:
        print("No retraining needed.")
        return None
    response = sagemaker_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName="retrain-on-drift",
        PipelineExecutionDescription="; ".join(decision.reasons)[:3000],
    )
    print("Retraining started:", response["PipelineExecutionArn"])
    return response["PipelineExecutionArn"]


def create_alarm_and_rule(cloudwatch_client, events_client, pipeline_arn: str, role_arn: str,
                          metric_name: str = "drifted_features") -> None:
    """Alarm when any feature drifts; EventBridge starts the pipeline on ALARM."""
    alarm = "steam-tag-gate-drift"
    cloudwatch_client.put_metric_alarm(
        AlarmName=alarm, Namespace=CLOUDWATCH_NAMESPACE, MetricName=metric_name,
        Dimensions=[{"Name": "Model", "Value": "tag-gate"}],
        Statistic="Maximum", Period=300, EvaluationPeriods=1, Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold", TreatMissingData="notBreaching",
        AlarmDescription="Tag QA Gate input drift -> retrain",
    )
    rule = "steam-tag-gate-retrain-on-drift"
    events_client.put_rule(
        Name=rule,
        EventPattern=json.dumps({
            "source": ["aws.cloudwatch"],
            "detail-type": ["CloudWatch Alarm State Change"],
            "detail": {"alarmName": [alarm], "state": {"value": ["ALARM"]}},
        }),
        State="ENABLED",
    )
    events_client.put_targets(Rule=rule, Targets=[{
        "Id": "tag-gate-pipeline", "Arn": pipeline_arn, "RoleArn": role_arn,
        "SageMakerPipelineParameters": {"PipelineParameterList": []},
    }])
