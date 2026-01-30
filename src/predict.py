import pandas as pd
import pickle

# Load models
with open('models/rf.pkl', 'rb') as f:
    rf = pickle.load(f)
with open('models/xgb.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Load data
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)

# Predict
rf_proba = rf.predict_proba(X)[:, 1]
xgb_proba = xgb_model.predict_proba(X)[:, 1]
ensemble_proba = (rf_proba + xgb_proba) / 2
predictions = (ensemble_proba > 0.5).astype(int)

# Save results
results = pd.DataFrame({
    'fraud_probability': ensemble_proba,
    'prediction': predictions
})
results.to_csv('results/tables/predictions.csv', index=False)

print(f"Predictions saved")
print(f"Detected {predictions.sum()} fraudulent transactions")
