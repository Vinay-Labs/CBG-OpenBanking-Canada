import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import openai
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIG & SECRETS
# ==========================================
st.set_page_config(page_title="AI Underwriting Engine", page_icon="🏦", layout="wide")

try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai.api_key = "YOUR_API_KEY_HERE"

# ==========================================
# 2. DATA DICTIONARY (Aligns with Notebook & SQL)
# ==========================================
OFFICIAL_DICTIONARY = {
    "credit_to_income_ratio": "Credit-to-Income Ratio",
    "annuity_to_income_ratio": "Annuity-to-Income Ratio",
    "active_loan_ratio": "Active Loan Ratio",
    "total_previous_loans": "Total Previous Loans",
    "avg_credit_sum": "Avg Previous Credit Amount",
    "bureau_debt_to_credit_ratio": "Bureau Debt-to-Credit Ratio",
    "total_installments": "Total Installment Count",
    "avg_delay": "Average Payment Delay (Days)",
    "late_ratio": "Late Payment Ratio",
    "payment_ratio": "Payment-to-Installment Ratio",
    "EXT_SOURCE_1": "Alt Proxy 1 (Telco)",
    "EXT_SOURCE_2": "Alt Proxy 2 (Cash Flow)",
    "EXT_SOURCE_3": "Alt Proxy 3 (Behavioral)",
    "EXT_SOURCE_MEAN": "Aggregated Alt Score",
    "EXT_SOURCE_PRODUCT": "Alt Score Reliability Index",
    "AGE": "Applicant Age",
    "DAYS_EMPLOYED": "Employment Duration",
    "AMT_CREDIT": "Loan Amount (Log)",
    "AMT_INCOME_TOTAL": "Total Income"
}

# ==========================================
# 3. AI GENERATION ENGINE
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision):
    prompt = f"""
    You are an expert AI Underwriting Assistant. 
    Decision: {decision} | Default Probability: {probability:.2%}
    
    SHAP EVIDENCE (Math-based):
    {shap_evidence}
    
    TASK: Provide an INTERNAL MEMO (Technical) and an APPLICANT LETTER (Gentle/Plain English).
    Translate technical features using the descriptions provided. 
    A positive impact (+) increases risk; a negative impact (-) decreases risk.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a professional bank underwriter."},
                      {"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Report Generation Error: {str(e)}"

# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI & RAW INPUTS (With Units)
# ==========================================
st.title("🏦 FinTech AI: Alternative Data Underwriter")
st.markdown("Automated risk assessment for 'thin-file' applicants.")

st.sidebar.header("📝 Application Data")

# Sections with Units
st.sidebar.subheader("👤 Applicant Core")
age = st.sidebar.number_input("Age (Years)", 18, 100, 35)
days_emp = st.sidebar.number_input("Days Employed (Negative format)", value=-1500, help="Example: -1500 means 1500 days of history")
income = st.sidebar.number_input("Total Annual Income ($)", value=60000.0, step=1000.0)
credit = st.sidebar.number_input("Requested Loan Amount ($)", value=250000.0, step=5000.0)
annuity = st.sidebar.number_input("Monthly Payment / Annuity ($)", value=12500.0, step=500.0)

st.sidebar.subheader("🌐 Alternative Proxies (0.0 to 1.0)")
e1 = st.sidebar.slider("Alt Proxy 1 (Telco)", 0.0, 1.0, 0.5)
e2 = st.sidebar.slider("Alt Proxy 2 (Cash Flow)", 0.0, 1.0, 0.6)
e3 = st.sidebar.slider("Alt Proxy 3 (Behavioral)", 0.0, 1.0, 0.5)

st.sidebar.subheader("📊 Bureau & Installment History")
prev_loans = st.sidebar.number_input("Total Previous Loans", 0, 50, 3)
active_loans = st.sidebar.number_input("Active Loans", 0, 50, 1)
b_debt_ratio = st.sidebar.slider("Bureau Debt-to-Credit Ratio (%)", 0.0, 1.0, 0.3)
tot_inst = st.sidebar.number_input("Total Past Installments", 0, 500, 24)
l_ratio = st.sidebar.slider("Late Payment Ratio (%)", 0.0, 1.0, 0.05)
a_delay = st.sidebar.number_input("Avg Payment Delay (Days, + is late)", value=-2.0)
p_ratio = st.sidebar.slider("Payment Ratio (Amt Paid / Amt Due)", 0.0, 2.0, 1.0)

# ==========================================
# 6. ENGINE: CALCULATE & PREDICT
# ==========================================
if st.button("🚀 Evaluate Applicant"):
    
    # 1. SQL-Aligned Feature Engineering
    ext_mean = (e1 + e2 + e3) / 3
    ext_prod = e1 * e2 * e3
    c_to_i = credit / income if income > 0 else 0
    a_to_i = annuity / income if income > 0 else 0
    act_ratio = active_loans / prev_loans if prev_loans > 0 else 0
    
    # Skewness Transform (Phase 4)
    amt_credit_log = np.log1p(credit)

    # 2. Map to Model Features
    input_dict = {
        'AGE': age, 'DAYS_EMPLOYED': days_emp, 'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': amt_credit_log, 'AMT_ANNUITY': annuity,
        'EXT_SOURCE_1': e1, 'EXT_SOURCE_2': e2, 'EXT_SOURCE_3': e3,
        'EXT_SOURCE_MEAN': ext_mean, 'EXT_SOURCE_PRODUCT': ext_prod,
        'total_previous_loans': prev_loans, 'active_loans': active_loans,
        'bureau_debt_to_credit_ratio': b_debt_ratio,
        'total_installments': tot_inst, 'late_ratio': l_ratio,
        'avg_delay': a_delay, 'payment_ratio': p_ratio,
        'credit_to_income_ratio': c_to_i, 'annuity_to_income_ratio': a_to_i,
        'active_loan_ratio': act_ratio
    }

    # Robust Feature Extraction (Fixed the NoneType error)
    if hasattr(model, 'feature_names_in_'): model_features = model.feature_names_in_
    elif hasattr(model, 'feature_name_'): model_features = model.feature_name_
    else: model_features = model.booster_.feature_name()

    # Create 1-row DataFrame horizontally
    full_input = {feat: [input_dict.get(feat, 0.0)] for feat in model_features}
    input_df = pd.DataFrame(full_input)

    # 3. Decision
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    st.subheader(f"Decision: {'❌' if decision=='REJECTED' else '✅'} {decision}")
    st.progress(prob)
    st.write(f"Default Risk Probability: **{prob:.2%}**")

    # 4. SHAP Logic
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    # LightGBM logic check
    shap_vals_class1 = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    
    top_idx = np.argsort(np.abs(shap_vals_class1))[-5:]
    shap_evidence = ""
    for i in top_idx:
        f_name = input_df.columns[i]
        impact = shap_vals_class1[i]
        desc = OFFICIAL_DICTIONARY.get(f_name, "Internal Metric")
        shap_evidence += f"- {f_name} ({desc}): {impact:+.4f} risk impact\n"

    # 5. LLM Report
    with st.spinner("AI is analyzing SHAP values..."):
        report = generate_ai_feedback(prob, shap_evidence, decision)
        st.divider()
        st.markdown(report)
