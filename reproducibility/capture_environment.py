from pathlib import Path
from .capture import capture_environment
if __name__ == '__main__':
    capture_environment('results/reproducibility/environment', Path(__file__).resolve().parents[1])
