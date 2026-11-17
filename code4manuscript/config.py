"""Input and output roots; override with environment variables if needed."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("MANUSCRIPT_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUTPUT_ROOT = Path(os.environ.get("MANUSCRIPT_OUTPUT_ROOT", PROJECT_ROOT / "manuscript_code_outputs")).resolve()
CLASSIFIER_WORK = OUTPUT_ROOT / "classifier"
POLICY_WORK = OUTPUT_ROOT / "policy_work"
