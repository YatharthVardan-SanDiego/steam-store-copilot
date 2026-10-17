"""Helper: build the pipeline definition without AWS (all S3 uploads stubbed)."""
import json
from contextlib import ExitStack
from unittest import mock

import sagemaker


def build_definition() -> dict:
    up = mock.Mock(s3_prefix="s3://bkt/code/sourcedir.tar.gz", script_name="train.py")
    patches = [
        mock.patch.object(sagemaker.session.Session, "default_bucket", lambda self: "bkt"),
        mock.patch.object(sagemaker.session.Session, "upload_data",
                          lambda self, path, bucket=None, key_prefix="data", **kw:
                          f"s3://{bucket or 'bkt'}/{key_prefix}/x"),
        mock.patch.object(sagemaker.session.Session, "upload_string_as_file_body",
                          lambda self, body, bucket, key, **kw: f"s3://{bucket}/{key}"),
        mock.patch("sagemaker.estimator.tar_and_upload_dir", lambda *a, **k: up),
        mock.patch("sagemaker.fw_utils.tar_and_upload_dir", lambda *a, **k: up),
        mock.patch("sagemaker.s3.S3Uploader.upload", lambda *a, **k: "s3://bkt/code/x"),
        mock.patch("sagemaker.s3.S3Uploader.upload_string_as_file_body",
                   lambda *a, **k: "s3://bkt/code/runproc.sh"),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        from src.pipelines.tag_gate_pipeline import get_pipeline

        pipeline = get_pipeline(role="arn:aws:iam::123456789012:role/LabRole",
                                bucket="bkt", region="us-east-1")
        return json.loads(pipeline.definition())
