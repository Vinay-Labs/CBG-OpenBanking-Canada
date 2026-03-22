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
st.set_page_config(page_title="Nexus AI Underwriting", page_icon="🛡️", layout="wide")

# Securely load OpenAI Key from Streamlit Secrets
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("⚠️ OpenAI API Key not found in Secrets. Please add it to 'Advanced Settings' in Streamlit.")

# ==========================================
# 2. GLOBAL DATA DICTIONARY (For Tooltips & AI)
# ==========================================
OFFICIAL_DICTIONARY = {
    "EXT_SOURCE_1": "External Bureau Score 1 (e.g., Telecom/Utility history)",
    "EXT_SOURCE_2": "External Bureau Score 2 (e.g., Credit Bureau/Banking history)",
    "EXT_SOURCE_3": "External Bureau Score 3 (e.g., Alternative Behavioral data)",
    "EXT_SOURCE_MEAN": "Aggregated Average of all External Sources",
    "EXT_SOURCE_PRODUCT": "Product of all External Sources (Reliability Index)",
    "late_ratio": "Percentage of past payments that were made after the due date",
    "payment_ratio": "Total Amount Paid vs. Total Amount Due (Lower than 1.0 is risky)",
    "avg_delay": "Average number of days payments were delayed (Positive = Late)",
    "credit_to_income_ratio": "Loan Amount relative to Annual Income",
    "annuity_to_income_ratio": "Monthly Payment relative to Annual Income",
    "active_loan_ratio": "Proportion of currently active loans vs. total history",
    "AMT_CREDIT": "Total Loan Amount requested (Log Transformed)",
    "AGE": "Applicant's age in years",
    "DAYS_EMPLOYED": "Duration of current employment (Days)",
    "AMT_INCOME_TOTAL": "Gross Annual Income"
}

# ==========================================
# 3. ROBUST AI GENERATION ENGINE
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision):
    prompt = f"""
    You are a Senior Risk Underwriter. A loan application was processed by our LightGBM model.
    Decision: {decision} | Default Risk Probability: {probability:.2%}
    
    SHAP MATH EVIDENCE:
    {shap_evidence}
    
    TASK: Provide a technical memo for the bank and a professional letter for the applicant.
    - Translate technical names like 'late_ratio' to 'history of payment delays'.
    - Ensure 'Positive Impact' is interpreted as 'Increased Risk' and 'Negative' as 'Decreased Risk'.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an expert bank underwriter."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM Offline. Decision: {decision} based on risk score {probability:.2%}."

# ==========================================
# 4. LOAD MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI: NEXUS AI UNDERWRITING SUITE
# ==========================================
st.title("🛡️ Nexus AI: Cognitive Credit Underwriting Suite")
st.markdown("#### *High-Precision Default Risk Prediction & Explainable AI (XAI)*")
st.divider()

st.sidebar.header("📊 Application Dossier")

# --- SIDEBAR: FINANCIALS ---
with st.sidebar.expander("💰 Core Financials", expanded=True):
    income = st.number_input("Annual Income ($)", value=60000.0, help=OFFICIAL_DICTIONARY["AMT_INCOME_TOTAL"])
    credit_amt = st.number_input("Requested Loan ($)", value=250000.0, help=OFFICIAL_DICTIONARY["AMT_CREDIT"])
    annuity = st.number_input("Monthly Installment ($)", value=12500.0, help="Monthly payment amount.")
    age = st.number_input("Applicant Age (Years)", 18, 90, 35, help=OFFICIAL_DICTIONARY["AGE"])
    days_emp = st.number_input("Days Employed (Negative)", value=-1500, help=OFFICIAL_DICTIONARY["DAYS_EMPLOYED"])

# --- SIDEBAR: EXT SOURCES ---
with st.sidebar.expander("🌐 External Risk Scores (EXT)", expanded=True):
    e1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, help=OFFICIAL_DICTIONARY["EXT_SOURCE_1"])
    e2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.6, help=OFFICIAL_DICTIONARY["EXT_SOURCE_2"])
    e3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.4, help=OFFICIAL_DICTIONARY["EXT_SOURCE_3"])

# --- SIDEBAR: HISTORY WORKSHEET ---
with st.sidebar.expander("📝 History Worksheet (Auto-Calc)", expanded=False):
    st.caption("AI automatically calculates ratios based on raw entry.")
    tot_due = st.number_input("Total Amount Due ($)", value=10000.0)
    tot_paid = st.number_input("Total Amount Paid ($)", value=9800.0)
    p_ratio = tot_paid / tot_due if tot_due > 0 else 1.0
    
    l_count = st.number_input("Late Payment Count", 0, 500, 2)
    t_inst = st.number_input("Total Installments", 1, 500, 24)
    l_ratio = l_count / t_inst
    
    a_delay = st.number_input("Avg Payment Delay (Days)", value=2.5, help=OFFICIAL_DICTIONARY["avg_delay"])
    
    p_loans = st.number_input("Total Previous Loans", 0, 50, 3)
    a_loans = st.number_input("Active Loans", 0, 50, 1)
    act_ratio = a_loans / p_loans if p_loans > 0 else 0
    b_debt = st.slider("Bureau Debt-to-Credit (%)", 0.0, 1.0, 0.3, help=OFFICIAL_DICTIONARY["bureau_debt_to_credit_ratio"])

# ==========================================
# 6. EXECUTION ENGINE
# ==========================================
if st.button("🚀 EXECUTE RISK ASSESSMENT", type="primary"):
    
    # 1. Feature Engineering
    ext_mean = (e1 + e2 + e3) / 3
    ext_prod = e1 * e2 * e3
    c_to_i = credit_amt / income if income > 0 else 0
    a_to_i = annuity / income if income > 0 else 0
    
    # Map to model columns
    raw_data = {
        'EXT_SOURCE_1': e1, 'EXT_SOURCE_2': e2, 'EXT_SOURCE_3': e3,
        'EXT_SOURCE_MEAN': ext_mean, 'EXT_SOURCE_PRODUCT': ext_prod,
        'AGE': age, 'DAYS_EMPLOYED': days_emp, 'AMT_INCOME_TOTAL': income, 
        'AMT_CREDIT': np.log1p(credit_amt), 'AMT_ANNUITY': annuity, 
        'payment_ratio': p_ratio, 'late_ratio': l_ratio, 'avg_delay': a_delay,
        'credit_to_income_ratio': c_to_i, 'annuity_to_income_ratio': a_to_i,
        'active_loan_ratio': act_ratio, 'total_previous_loans': p_loans,
        'active_loans': a_loans, 'bureau_debt_to_credit_ratio': b_debt
    }

    # Robust Feature Alignment
    if hasattr(model, 'feature_names_in_'): model_features = model.feature_names_in_
    elif hasattr(model, 'feature_name_'): model_features = model.feature_name_
    else: model_features = model.booster_.feature_name()

    # Create 1-row DataFrame (HORIZONTAL FIX)
    input_df = pd.DataFrame({feat: [raw_data.get(feat, 0.0)] for feat in model_features})

    # 2. Predict
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    st.divider()
    col_res1, col_res2 = st.columns(2)
    with col_res
