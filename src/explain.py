import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load models
with open('models/rf.pkl', 'rb') as f:
    rf = pickle.load(f)

# Load data
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)

# Sample for SHAP
X_sample = X.sample(n=100, random_state=42)

# Generate SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Summary plot
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig('results/figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("SHAP explanations generated")
print("Summary plot saved to results/figures/shap_summary.png")
