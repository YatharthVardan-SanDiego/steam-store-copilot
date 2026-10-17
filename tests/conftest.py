import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# model_lib / inference are imported flat inside the SageMaker container.
sys.path.insert(0, os.path.join(ROOT, "src", "models", "tag_gate"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
