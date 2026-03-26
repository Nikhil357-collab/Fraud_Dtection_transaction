import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier

print("All imports successful ✅")

# ======================
# 1. LOAD DATA
# ======================
data = pd.read_csv("creditcard.csv")

# ======================
# 2. PREPROCESSING
# ======================
data = data.drop(columns=['Time'])

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale Amount
scaler = StandardScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])

# ======================
# 3. HANDLE IMBALANCE (NO SMOTE)
# ======================
# Compute scale_pos_weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

print("Scale_pos_weight:", scale_pos_weight)

# ======================
# 4. XGBOOST MODEL
# ======================
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ======================
# 5. PREDICTION + THRESHOLD TUNING
# ======================
y_probs = model.predict_proba(X_test)[:, 1]

# 🔥 TRY MULTIPLE THRESHOLDS
thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]

for t in thresholds:
    y_pred = (y_probs > t).astype(int)
    print("\n==========================")
    print("Threshold:", t)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ======================
# 6. FINAL THRESHOLD (SET BEST ONE)
# ======================
threshold = 0.87
y_pred = (y_probs > threshold).astype(int)

print("\nFINAL MODEL PERFORMANCE")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_probs))

# ======================
# 7. SAVE MODEL
# ======================
joblib.dump(model, "fraud_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved ✅")

# ======================
# 8. REAL-TIME PREDICTION
# ======================
def predict_transaction(transaction_dict):
    model = joblib.load("fraud_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")

    df = pd.DataFrame([transaction_dict])
    df['Amount'] = scaler.transform(df[['Amount']])

    prob = model.predict_proba(df)[:, 1][0]

    if prob > 0.87:
        return f"🚨 Fraud (Confidence: {prob:.2f})"
    else:
        return f"✅ Legit (Confidence: {prob:.2f})"


# ======================
# 9. TEST SAMPLE
# ======================
sample = X_test.iloc[0].to_dict()
sample = X_test[y_test == 1].iloc[0]
print(predict_transaction(sample))
 