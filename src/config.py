"""Shared names and S3 layout for every notebook, pipeline and script."""

from __future__ import annotations

from dataclasses import dataclass

PROJECT = "steam-store-copilot"

DRAFT_FEATURE_GROUP = "steam-draft-features"
POST_RELEASE_FEATURE_GROUP = "steam-post-release-features"

TAG_GATE_MODEL_GROUP = "steam-tag-gate"
RECEPTION_MODEL_GROUP = "steam-reception-estimator"
PIPELINE_NAME = "steam-tag-gate-pipeline"

CLOUDWATCH_NAMESPACE = "SteamStoreCopilot"
DASHBOARD_NAME = "SteamStoreCopilot"

SKLEARN_VERSION = "1.2-1"


@dataclass(frozen=True)
class S3Layout:
    bucket: str
    prefix: str = PROJECT

    def uri(self, *parts: str) -> str:
        return "s3://" + "/".join([self.bucket, self.prefix, *parts])

    @property
    def raw(self) -> str:
        return self.uri("raw")

    @property
    def curated(self) -> str:
        return self.uri("curated")

    @property
    def snapshot(self) -> str:
        return self.uri("curated", "tag_gate_snapshot")

    @property
    def offline_store(self) -> str:
        return self.uri("feature-store")

    @property
    def pipeline_artifacts(self) -> str:
        return self.uri("pipeline")

    @property
    def batch(self) -> str:
        return self.uri("batch")

    @property
    def monitoring(self) -> str:
        return self.uri("monitoring")


def session_layout(sagemaker_session=None) -> S3Layout:
    """S3 layout rooted at the SageMaker default bucket of the current account."""
    import sagemaker

    sess = sagemaker_session or sagemaker.Session()
    return S3Layout(bucket=sess.default_bucket())
