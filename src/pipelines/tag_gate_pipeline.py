"""SageMaker Pipeline for the Tag QA Gate.

    PrepareData ─► TrainBaseline ─► TrainCandidate ─► Evaluate ─► QualityGate
                                                                   ├─ pass ─► RegisterTagGate (Model Registry)
                                                                   └─ fail ─► QualityGateFailed (FailStep)

The quality gate requires the candidate to (a) reach ``MacroF1Threshold`` on the
held-out test split and (b) beat the TF-IDF baseline. Running the pipeline with
an unreachable threshold (e.g. 0.95) demonstrates the failed state.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from src.config import PIPELINE_NAME, SKLEARN_VERSION, TAG_GATE_MODEL_GROUP, S3Layout

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG_GATE_DIR = os.path.join(ROOT, "src", "models", "tag_gate")
SPLITS = ["train", "val", "test", "incoming", "monitor"]

METRIC_DEFINITIONS = [
    {"Name": "val:macro_f1", "Regex": r"val_macro_f1=([0-9\.]+);"},
    {"Name": "val:micro_f1", "Regex": r"val_micro_f1=([0-9\.]+);"},
    {"Name": "val:auto_approval", "Regex": r"val_auto_approval=([0-9\.]+);"},
    {"Name": "val:false_approval", "Regex": r"val_false_approval=([0-9\.]+);"},
]


def stage_processing_code() -> str:
    """Flat directory with the processing scripts and the modules they import."""
    out = tempfile.mkdtemp(prefix="tag-gate-code-")
    for src in (os.path.join(ROOT, "src", "pipelines", "scripts", "prepare.py"),
                os.path.join(ROOT, "src", "pipelines", "scripts", "evaluate.py"),
                os.path.join(ROOT, "src", "data", "preprocess.py"),
                os.path.join(TAG_GATE_DIR, "model_lib.py")):
        shutil.copy(src, out)
    return out


def get_pipeline(role: str, bucket: str | None = None, region: str | None = None,
                 pipeline_name: str = PIPELINE_NAME) -> Pipeline:
    boto_session = boto3.Session(region_name=region) if region else boto3.Session()
    session = PipelineSession(boto_session=boto_session, default_bucket=bucket)
    layout = S3Layout(bucket=session.default_bucket())
    code_dir = stage_processing_code()

    # --- Parameters -------------------------------------------------------------
    snapshot_uri = ParameterString("SnapshotUri", default_value=layout.snapshot)
    processing_instance = ParameterString("ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance = ParameterString("TrainingInstanceType", default_value="ml.m5.xlarge")
    f1_threshold = ParameterFloat("MacroF1Threshold", default_value=0.55)
    max_false_approval = ParameterFloat("MaxFalseApproval", default_value=0.05)
    approval_status = ParameterString("ModelApprovalStatus",
                                      default_value="PendingManualApproval")

    def run_prefix(*parts: str):
        return Join(on="/", values=[layout.pipeline_artifacts,
                                    ExecutionVariables.PIPELINE_EXECUTION_ID, *parts])

    processor = FrameworkProcessor(
        estimator_cls=SKLearn, framework_version=SKLEARN_VERSION, py_version="py3",
        role=role, instance_type=processing_instance, instance_count=1,
        base_job_name="tag-gate-processing", sagemaker_session=session,
    )

    # --- 1. Prepare / validate data ---------------------------------------------
    step_prepare = ProcessingStep(
        name="PrepareData",
        step_args=processor.run(
            code="prepare.py", source_dir=code_dir,
            inputs=[ProcessingInput(source=snapshot_uri,
                                    destination="/opt/ml/processing/input/snapshot")],
            outputs=[ProcessingOutput(output_name=s, source=f"/opt/ml/processing/output/{s}",
                                      destination=run_prefix("data", s)) for s in SPLITS],
        ),
    )
    outputs = step_prepare.properties.ProcessingOutputConfig.Outputs
    channels = {
        "train": TrainingInput(outputs["train"].S3Output.S3Uri, content_type="text/csv"),
        "val": TrainingInput(outputs["val"].S3Output.S3Uri, content_type="text/csv"),
    }

    # --- 2/3. Train baseline and candidate --------------------------------------
    def training_step(kind: str, name: str, depends_on=None) -> TrainingStep:
        estimator = SKLearn(
            entry_point="train.py", source_dir=TAG_GATE_DIR,
            framework_version=SKLEARN_VERSION, py_version="py3", role=role,
            instance_type=training_instance, instance_count=1,
            hyperparameters={"kind": kind, "max-false-approval": max_false_approval},
            metric_definitions=METRIC_DEFINITIONS, base_job_name=f"tag-gate-{kind}",
            output_path=layout.uri("models", kind), sagemaker_session=session,
        )
        return TrainingStep(name=name, step_args=estimator.fit(inputs=channels),
                            depends_on=depends_on)

    step_baseline = training_step("baseline", "TrainBaseline")
    # Sequential to stay inside Learner Lab's concurrent-job limits.
    step_candidate = training_step("improved", "TrainCandidate", depends_on=[step_baseline])

    # --- 4. Evaluate on held-out test --------------------------------------------
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation",
                                     path="evaluation.json")
    step_evaluate = ProcessingStep(
        name="Evaluate",
        step_args=processor.run(
            code="evaluate.py", source_dir=code_dir,
            inputs=[
                ProcessingInput(source=step_baseline.properties.ModelArtifacts.S3ModelArtifacts,
                                destination="/opt/ml/processing/model/baseline"),
                ProcessingInput(source=step_candidate.properties.ModelArtifacts.S3ModelArtifacts,
                                destination="/opt/ml/processing/model/candidate"),
                ProcessingInput(source=outputs["test"].S3Output.S3Uri,
                                destination="/opt/ml/processing/test"),
            ],
            outputs=[ProcessingOutput(output_name="evaluation",
                                      source="/opt/ml/processing/evaluation",
                                      destination=run_prefix("evaluation"))],
        ),
        property_files=[evaluation_report],
    )

    # --- 5. Quality gate -> register or fail --------------------------------------
    evaluation_uri = Join(on="/", values=[
        step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        "evaluation.json"])
    model = SKLearnModel(
        model_data=step_candidate.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point="inference.py", source_dir=TAG_GATE_DIR,
        framework_version=SKLEARN_VERSION, py_version="py3", role=role,
        sagemaker_session=session,
    )
    step_register = ModelStep(
        name="RegisterTagGate",
        step_args=model.register(
            content_types=["application/jsonlines"],
            response_types=["application/jsonlines"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=TAG_GATE_MODEL_GROUP,
            approval_status=approval_status,
            model_metrics=ModelMetrics(model_statistics=MetricsSource(
                s3_uri=evaluation_uri, content_type="application/json")),
            description="Tag QA Gate: TF-IDF (word+char) + logistic regression, 22 tags",
        ),
    )
    step_fail = FailStep(
        name="QualityGateFailed",
        error_message=Join(on=" ", values=[
            "Candidate rejected: test macro-F1 below", f1_threshold,
            "or no lift over the TF-IDF baseline."]),
    )
    step_gate = ConditionStep(
        name="QualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step_name=step_evaluate.name, property_file=evaluation_report,
                             json_path="tag_gate.macro_f1.value"),
                right=f1_threshold),
            ConditionGreaterThan(
                left=JsonGet(step_name=step_evaluate.name, property_file=evaluation_report,
                             json_path="tag_gate.lift_over_baseline.value"),
                right=0.0),
        ],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[snapshot_uri, processing_instance, training_instance, f1_threshold,
                    max_false_approval, approval_status],
        steps=[step_prepare, step_baseline, step_candidate, step_evaluate, step_gate],
        sagemaker_session=session,
    )


def default_role() -> str:
    """Execution role inside SageMaker Studio (LabRole in AWS Academy)."""
    try:
        return sagemaker.get_execution_role()
    except ValueError:
        account = boto3.client("sts").get_caller_identity()["Account"]
        return f"arn:aws:iam::{account}:role/LabRole"
