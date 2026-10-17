"""The pipeline definition compiles offline (no AWS calls) with the expected DAG."""
import os

import pytest

pytest.importorskip("sagemaker")


@pytest.fixture(scope="module")
def definition():
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    from tests.pipeline_offline import build_definition

    return build_definition()


def test_steps_and_quality_gate(definition):
    names = [s["Name"] for s in definition["Steps"]]
    assert names == ["PrepareData", "TrainBaseline", "TrainCandidate", "Evaluate", "QualityGate"]
    gate = definition["Steps"][-1]["Arguments"]
    assert [s["Type"] for s in gate["IfSteps"]] == ["RegisterModel"]
    assert [s["Type"] for s in gate["ElseSteps"]] == ["Fail"]


def test_parameters(definition):
    params = {p["Name"]: p.get("DefaultValue") for p in definition["Parameters"]}
    assert params["MacroF1Threshold"] == 0.55
    assert params["ModelApprovalStatus"] == "PendingManualApproval"
