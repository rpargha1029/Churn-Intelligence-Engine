import shap
import joblib
import pandas as pd




def explain_model(model_path, X_sample, out_html=None):
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, show=False)
if out_html:
import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig(out_html, bbox_inches='tight')
return shap_values