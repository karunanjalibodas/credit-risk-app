import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="AI Credit Risk System", layout="centered")

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            height: 100%;
            background: linear-gradient(135deg, #0f172a, #1e293b);
        }
        .login-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
        }
        .login-card {
            width: 350px;
            padding: 35px;
            border-radius: 15px;
            background: transparent;
            text-align: center;
        }
        .title-main {
            font-size: 20px;
            font-weight: bold;
            color: white;
            margin-bottom: 20px;
        }
        .login-title {
            font-size: 24px;
            font-weight: bold;
            color: white;
            margin-bottom: 15px;
        }
        .stTextInput>div>div>input {
            background-color: #374151;
            color: white;
            border-radius: 8px;
        }
        .stButton>button {
            background-color: #22c55e;
            color: white;
            border-radius: 10px;
            height: 45px;
            width: 100%;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    # ✅ TITLE ADDED HERE
    st.markdown("""
        <div class="title-main">
        💳 AI-Powered Credit Risk Scoring System
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-title">🔐 Login</div>', unsafe_allow_html=True)

    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", key="login_btn"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ---------------- MAIN APP ----------------

st.title("💳 AI-Powered Credit Risk Scoring System")
st.markdown("### Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, key="age")
    sex = st.selectbox("Sex", ["male", "female"], key="sex")
    job = st.selectbox("Job Level (0-3)", [0, 1, 2, 3], key="job")
    housing = st.selectbox("Housing", ["own", "rent", "free"], key="housing")

with col2:
    saving = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "unknown"], key="saving")
    checking = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"], key="checking")
    credit = st.number_input("Credit Amount", min_value=100.0, key="credit")
    duration = st.number_input("Duration (months)", min_value=1.0, key="duration")

purpose = st.selectbox("Purpose", ["car", "furniture", "radio/TV", "education", "business"], key="purpose")

# Validation
if credit <= 0 or duration <= 0:
    st.warning("⚠️ Please enter valid values")
    st.stop()

# Create DataFrame
data = pd.DataFrame([[age, sex, job, housing, saving, checking, credit, duration, purpose]],
columns=["Age","Sex","Job","Housing","Saving accounts","Checking account","Credit amount","Duration","Purpose"])

# Encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in data.select_dtypes(include="object"):
    data[col] = le.fit_transform(data[col])

# Prediction
if st.button("🔍 Predict Risk", key="predict_btn"):

    prediction = model.predict(data)[0]

    st.subheader("📊 Result")

    if prediction == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")

    st.metric("Risk Status", "High Risk" if prediction == 1 else "Low Risk")

    # Recommendations
    st.subheader("💡 Recommendations")

    suggestions = []

    if credit > 5000:
        suggestions.append("Reduce loan amount")

    if duration > 24:
        suggestions.append("Reduce loan duration")

    if job == 0:
        suggestions.append("Improve job stability")

    if len(suggestions) == 0:
        st.success("✔️ Profile looks good")

    for s in suggestions:
        st.write("👉", s)

    # SHAP
    st.subheader("📊 Explainable AI (SHAP)")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_val = shap_array[0][:, 1]
        elif shap_array.ndim == 2:
            shap_val = shap_array[0]
        elif isinstance(shap_values, list):
            shap_val = shap_values[1][0]
        else:
            shap_val = shap_array.flatten()

        shap_df = pd.DataFrame([shap_val], columns=data.columns)

        st.write("Feature Contributions:")
        st.dataframe(shap_df)

        st.subheader("📈 Feature Impact Visualization")
        st.bar_chart(shap_df.T)

    except Exception as e:
        st.error(f"SHAP error: {e}")
