import joblib
import os


def save_model(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: str):
    return joblib.load(path)
