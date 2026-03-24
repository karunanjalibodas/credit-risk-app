# ---------------- MAIN APP ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Session step
if "step" not in st.session_state:
    st.session_state.step = 1

# ---------------- STEP 1 ----------------
if st.session_state.step == 1:

    st.markdown('<p class="big-title">💳 Get Instant Loan in Minutes</p>', unsafe_allow_html=True)
    st.markdown("### AI Powered • No Paperwork • Fast Approval")

    if st.button("🚀 Start Application"):
        st.session_state.step = 2

# ---------------- STEP 2 ----------------
elif st.session_state.step == 2:

    st.subheader("📊 Basic Details")

    age = st.slider("Age", 18, 100)
    sex = st.radio("Gender", ["male", "female"])

    if st.button("Next"):
        st.session_state.age = age
        st.session_state.sex = sex
        st.session_state.step = 3

# ---------------- STEP 3 ----------------
elif st.session_state.step == 3:

    st.subheader("💰 Financial Details")

    job = st.slider("Job Level", 0, 3)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💸 Low Savings"):
            st.session_state.saving = "little"

    with col2:
        if st.button("💵 Medium"):
            st.session_state.saving = "moderate"

    with col3:
        if st.button("💎 High"):
            st.session_state.saving = "rich"

    housing = st.selectbox("Housing", ["own", "rent", "free"])

    if st.button("Next"):
        st.session_state.job = job
        st.session_state.housing = housing
        st.session_state.step = 4

# ---------------- STEP 4 ----------------
elif st.session_state.step == 4:

    st.subheader("🏦 Loan Details")

    credit = st.slider("💸 Loan Amount", 100, 20000, 5000)
    duration = st.slider("📅 Duration", 1, 48, 12)

    checking = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])
    purpose = st.selectbox("Purpose", ["car", "furniture", "radio/TV", "education", "business"])

    if st.button("🔍 Predict"):
        st.session_state.credit = credit
        st.session_state.duration = duration
        st.session_state.checking = checking
        st.session_state.purpose = purpose
        st.session_state.step = 5

# ---------------- STEP 5 RESULT ----------------
elif st.session_state.step == 5:

    st.subheader("📈 Your Credit Result")

    # Prepare data
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

    # Prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    score = int((1 - probability) * 1000)

    # -------- CREDIT SCORE --------
    st.markdown(f"""
    <div style="text-align:center;">
        <h1 style="font-size:60px; color:#00ffcc;">{score}</h1>
        <p>Credit Score</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Approval Chance", f"{int((1-probability)*100)}%")
    col2.metric("Risk Level", "Low Risk ✅" if prediction == 0 else "High Risk ❌")

    # -------- RECOMMENDATIONS --------
    st.subheader("💡 Recommendations")

    if prediction == 1:
        st.write("👉 Reduce loan amount")
        st.write("👉 Improve job stability")
        st.write("👉 Reduce duration")
    else:
        st.success("✔️ You are eligible 🎉")

    # -------- SHAP --------
    st.subheader("📊 AI Explanation")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_val = shap_array[0][:, 1]
        else:
            shap_val = shap_array[0]

        shap_df = pd.DataFrame([shap_val], columns=data.columns)

        st.bar_chart(shap_df.T)

    except Exception as e:
        st.error(f"SHAP error: {e}")

    if st.button("🔄 Start Again"):
        st.session_state.step = 1
