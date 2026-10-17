"""Upsert the Tag QA Gate pipeline and start an execution.

Used by the CD job in .github/workflows/ci.yml and by notebook 05.

    python -m scripts.run_pipeline                       # default parameters
    python -m scripts.run_pipeline --param MacroF1Threshold=0.95 --wait   # failing run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.tag_gate_pipeline import default_role, get_pipeline  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--param", action="append", default=[], help="Name=Value")
    p.add_argument("--definition-only", action="store_true")
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()

    role = args.role_arn or default_role()
    pipeline = get_pipeline(role=role, region=args.region)
    if args.definition_only:
        print(json.dumps(json.loads(pipeline.definition()), indent=2))
        return 0

    pipeline.upsert(role_arn=role, description="Tag QA Gate training pipeline")
    params = dict(kv.split("=", 1) for kv in args.param)
    execution = pipeline.start(parameters=params)
    print(f"Started {execution.arn}")
    if args.wait:
        try:
            execution.wait(delay=30, max_attempts=240)
        finally:
            for step in execution.list_steps():
                print(f"{step['StepName']:<22} {step['StepStatus']}")
        status = execution.describe()["PipelineExecutionStatus"]
        print(f"Pipeline status: {status}")
        return 0 if status == "Succeeded" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
