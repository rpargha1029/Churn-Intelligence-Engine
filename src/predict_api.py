import pandas as pd
import joblib
from src.preprocessing import clean_and_basic_process, encode_categoricals
from src.features import create_features




def predict_single(model_path: str, df: pd.DataFrame):
df = clean_and_basic_process(df)
df, enc = encode_categoricals(df)
df = create_features(df)
df = df.dropna()
model = joblib.load(model_path)
probs = model.predict_proba(df)[:,1]
df['churn_probability'] = probs
return df