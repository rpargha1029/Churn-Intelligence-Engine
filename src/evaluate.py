import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib




def load_and_eval(model_path, X_test, y_test):
model = joblib.load(model_path)
probs = model.predict_proba(X_test)[:,1]
preds = model.predict(X_test)
cm = confusion_matrix(y_test, preds)
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
return {'confusion_matrix': cm, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}