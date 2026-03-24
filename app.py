# ---------------- IMPORTS ----------------
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Credit Risk System", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>

/* ---------- BACKGROUND ---------- */
.stApp {
    background: #f5f7fa;
}

/* ---------- CENTER LOGIN ---------- */
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 80vh;
}

/* ---------- LOGIN CARD ---------- */
.login-card {
    background: white;
    padding: 40px;
    border-radius: 20px;
    width: 350px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    text-align: center;
    animation: fadeIn 1s ease-in-out;
}

/* ---------- TITLE ---------- */
.title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* ---------- SUBTITLE ---------- */
.subtitle {
    color: gray;
    margin-bottom: 20px;
}

/* ---------- INPUT ---------- */
.stTextInput input {
    background-color: #ffffff !important;
    color: black !important;
    border-radius: 10px;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-weight: bold;
}

/* ---------- ANIMATION ---------- */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION INIT ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "step" not in st.session_state:
    st.session_state.step = 1

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    # LOGO (emoji based)
    st.markdown("## 💳")

    st.markdown('<div class="title">AI Credit Risk System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Secure Login</div>', unsafe_allow_html=True)

    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ---------------- LOGOUT ----------------
col1, col2 = st.columns([8,1])
with col2:
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# ---------------- STEP 1 ----------------
if st.session_state.step == 1:

    st.markdown("<h1 style='text-align:center;'>💳 Get Instant Loan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI Powered • Fast Approval</p>", unsafe_allow_html=True)

    if st.button("🚀 Start"):
        st.session_state.step = 2

# ---------------- STEP 2 ----------------
elif st.session_state.step == 2:

    st.subheader("📊 Basic Details")

    age = st.slider("Age", 18, 100, 25)
    sex = st.radio("Gender", ["male", "female"])

    if st.button("Next"):
        st.session_state.age = age
        st.session_state.sex = sex
        st.session_state.step = 3

# ---------------- STEP 3 ----------------
elif st.session_state.step == 3:

    st.subheader("💰 Financial Details")

    job = st.slider("Job Level", 0, 3, 1)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💸 Low"):
            st.session_state.saving = "little"

    with col2:
        if st.button("💵 Medium"):
            st.session_state.saving = "moderate"

    with col3:
        if st.button("💎 High"):
            st.session_state.saving = "rich"

    housing = st.selectbox("Housing", ["own", "rent", "free"])

    if st.button("Next"):
        if "saving" not in st.session_state:
            st.warning("Select savings type")
            st.stop()

        st.session_state.job = job
        st.session_state.housing = housing
        st.session_state.step = 4

# ---------------- STEP 4 ----------------
elif st.session_state.step == 4:

    st.subheader("🏦 Loan Details")

    credit = st.slider("Loan Amount", 100, 20000, 5000)
    duration = st.slider("Duration", 1, 48, 12)

    checking = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])
    purpose = st.selectbox("Purpose", ["car", "furniture", "radio/TV", "education", "business"])

    if st.button("Predict"):
        st.session_state.credit = credit
        st.session_state.duration = duration
        st.session_state.checking = checking
        st.session_state.purpose = purpose
        st.session_state.step = 5

# ---------------- RESULT ----------------
elif st.session_state.step == 5:

    required = ["age","sex","job","housing","saving","checking","credit","duration","purpose"]
    for r in required:
        if r not in st.session_state:
            st.warning("Complete all steps")
            st.session_state.step = 1
            st.stop()

    st.subheader("📈 Result")

    data = pd.DataFrame([[
        st.session_state.age,
        st.session_state.sex,
        st.session_state.job,
        st.session_state.housing,
        st.session_state.saving,
        st.session_state.checking,
        st.session_state.credit,
        st.session_state.duration,
        st.session_state.purpose
    ]],
    columns=["Age","Sex","Job","Housing","Saving accounts","Checking account","Credit amount","Duration","Purpose"])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in data.select_dtypes(include="object"):
        data[col] = le.fit_transform(data[col])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    score = int((1 - prob) * 1000)

    st.markdown(f"<h1 style='text-align:center;color:#0072ff;'>{score}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Credit Score</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Approval", f"{int((1-prob)*100)}%")
    c2.metric("Risk", "Low ✅" if pred==0 else "High ❌")

    st.subheader("💡 Suggestions")
    if pred == 1:
        st.write("👉 Reduce loan amount")
        st.write("👉 Improve job stability")
    else:
        st.success("Eligible 🎉")

    st.subheader("📊 SHAP")
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(data)
        shap_df = pd.DataFrame([shap_vals[0]], columns=data.columns)
        st.bar_chart(shap_df.T)
    except:
        st.warning("SHAP not supported")

    if st.button("Restart"):
        st.session_state.step = 1
