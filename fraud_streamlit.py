import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details:")

# Input fields (simplified)
amount = st.number_input("Transaction Amount", min_value=0.0)

# Dummy inputs for V1–V28 (you can improve later)
input_data = {}

for i in range(1, 29):
    input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

input_data["Amount"] = amount

#upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Scale Amount
    data["Amount"] = scaler.transform(data[["Amount"]])

    probs = model.predict_proba(data)[:, 1]
    data["Fraud_Probability"] = probs
    data["Prediction"] = (probs > 0.85).astype(int)

    st.write("### 📊 Prediction Results")
    st.dataframe(data.head())

    # Download button
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Results",
        csv,
        "fraud_predictions.csv",
        "text/csv"
    )
#
# Predict button
if st.button("Predict"):

    df = pd.DataFrame([input_data])
    
    df["Amount"] = scaler.transform(df[["Amount"]])
    prob = model.predict_proba(df)[:, 1][0]
    #define
    fraud_percent = prob * 100
    legit_percent = (1 - prob) * 100
    
    # 🔥 Risk Level
    if prob < 0.2:
        risk = "🟢 Low Risk"
    elif prob < 0.6:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    # 🔥 Display Results
    st.subheader("🔍 Prediction Result")

    if prob > 0.85:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Legit Transaction")

    st.write(f"💳 Fraud Probability: {fraud_percent:.4f}%")
    st.write(f"💰 Legit Probability: {legit_percent:.4f}%")
    st.write(f"⚠️ Risk Level: {risk}")
    st.progress(int(fraud_percent))
    #matplotlib
    import matplotlib.pyplot as plt

# Create bar chart
labels = ['Fraud', 'Legit']


fig, ax = plt.subplots()
ax.bar(labels, values)
ax.set_ylabel("Probability (%)")
ax.set_title("Fraud vs Legit Probability")

st.pyplot(fig)
# Scale amount
df["Amount"] = scaler.transform(df[["Amount"]])
prob = model.predict_proba(df)[:, 1][0]

if prob > 0.85:
       st.error(f"🚨 Fraud Detected! Confidence: {prob:.4f}")
else:  
    st.success(f"✅ Legit Transaction. Confidence: {prob:.4f}")