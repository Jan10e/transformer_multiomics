from pathlib import Path

# Define the root directory of the repository
REPO_ROOT = Path(__file__).resolve().parent.parent

# Define common paths
MODEL_PATH = REPO_ROOT / "models"
RESULT_PATH = REPO_ROOT / "results"
DATA_PATH = REPO_ROOT / "data"

# Prefix for dataset files
DATA_PREFIX = "20231023_092657_"