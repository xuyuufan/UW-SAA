"""Where the project's data and results live, relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_SET = ROOT / "data" / "caserta_840"     # 840 Caserta instances: only for final results
TRAINING_SET = ROOT / "data" / "training"    # 840 generated instances (seed 2026): for designing rules
REFERENCE_DATA = ROOT / "cpp_reference" / "reference_data"  # the authors' sample formulas
RESULTS = ROOT / "results"
