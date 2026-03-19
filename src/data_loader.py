import pandas as pd




def load_telco(path: str) -> pd.DataFrame:
"""Load Telco Customer Churn CSV into DataFrame."""
df = pd.read_csv(path)
return df