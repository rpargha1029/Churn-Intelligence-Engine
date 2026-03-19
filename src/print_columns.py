import pandas as pd

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
print(df.columns.tolist())
