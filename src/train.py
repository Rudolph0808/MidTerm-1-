import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import pickle

# Load data
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Fraud cases: {y_train.sum()} ({y_train.mean()*100:.3f}%)")

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight={0: 1.0, 1: 2.0},
    random_state=42
)
rf.fit(X_train, y_train)
print("\nRandom Forest trained")

# Train XGBoost
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    scale_pos_weight=scale_pos,
    reg_lambda=0.01,
    random_state=42
)
xgb_model.fit(X_train, y_train)
print("XGBoost trained")

# Evaluate ensemble
rf_proba = rf.predict_proba(X_test)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
ensemble_proba = (rf_proba + xgb_proba) / 2
ensemble_pred = (ensemble_proba > 0.5).astype(int)

print("\nEnsemble Performance:")
print(classification_report(y_test, ensemble_pred))

# Save models
with open('models/rf.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('models/xgb.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("Models saved")
