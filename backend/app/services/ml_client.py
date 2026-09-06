from pathlib import Path
import sys


ML_SRC = Path(__file__).resolve().parents[3] / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from predict import predict, predict_batch


def get_risk_prediction(work_dict: dict) -> dict:
    return predict(work_dict)


def get_risk_predictions(work_dicts: list[dict]) -> list[dict]:
    return predict_batch(work_dicts)
