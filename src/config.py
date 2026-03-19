import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


if not os.path.exists(MODELS_DIR):
os.makedirs(MODELS_DIR, exist_ok=True)