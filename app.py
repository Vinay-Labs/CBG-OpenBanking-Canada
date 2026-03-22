import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import google.generativeai as genai
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIG & GEMINI SETUP
# ==========================================
st.set_page_config(page_title="Nexus AI Underwriting", page_icon="🛡️", layout="wide")

# Securely load Google API Key
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("❌ 'GOOGLE_API_KEY' not found in Streamlit Secrets. Please check your settings.")

# ==========================================
# 2. DATA DICTIONARY
# ==========================================
OFFICIAL_DICTIONARY = {
    "EXT_SOURCE_1": "External Source 1: Alternative credit bureau data (Telco/Utilities).",
    "EXT_SOURCE_2": "External Source 2: Banking transaction and cash flow history.",
    "EXT_SOURCE_3": "External Source 3: Alternative behavioral digital markers.",
    "AMT_CREDIT": "Total requested loan amount.",
    "AMT_ANNUITY": "Monthly installment amount.",
    "AMT_INCOME_TOTAL": "Total gross annual income.",
    "AGE": "Applicant's age in years.",
    "DAYS_EMPLOYED": "Total days of current employment (Negative value).",
    "total_previous_loans": "Total number of previous credit applications.",
    "active_loans": "Number of currently active loans.",
    "avg_delay": "Average days late across historical payments.",
    "late_ratio": "Percentage of past installments paid after the due date.",
    "payment_ratio": "Amount Paid vs. Amount Due ratio.",
    "credit_to_income_ratio": "Loan amount relative to annual income.",
    "annuity_to_income_ratio": "Monthly payment relative to annual income.",
    "active_loan_ratio": "Proportion of previous loans that are still active."
}

# ==========================================
# 3. ROBUST GEMINI ENGINE
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision):
    # Prompt is moved inside the function
    prompt = f"""
    You are a Senior Credit Risk Underwriter.
    Decision: {decision} | Probability of Default: {probability:.2%}
    
    SHAP EVIDENCE:
    {shap_evidence}
    
    TASK: Provide a technical 'Internal Underwriting Memo' and a professional 'Applicant Letter'.
    - Use the descriptions provided in the evidence to explain risks.
    - Positive impact (+) means increased risk; Negative impact (-) means strength.
    """
    
    try:
        # We initialize inside the try block to catch any library errors
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        # If the API fails, show a clean explanation
        return f"""
        ### 🤖 Nexus AI: Manual Decision Summary (LLM Offline)
        **System Verdict:** {decision}
        **Risk Score:** {probability:.2%}
        
        **Top Risk Drivers (Calculated by SHAP):**
        {shap_evidence}
        
        *Note: AI Narrative is currently unavailable. Technical error: {str(e)}*
        """

# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.title("🛡️ Nexus AI: Cognitive Credit Underwriting Suite")
st.markdown("#### *High-Precision Default Risk Prediction & Explainable AI (XAI)*")
st.divider()

st.sidebar.header("📊 Application Dossier")

with st.sidebar.expander("💰 Core Financials", expanded=True):
    v_income = st.number_input("Annual Income ($)", value=60000.0)
    v_credit = st.number_input("Requested Loan ($)", value=250000.0)
    v_annuity = st.number_input("Monthly Installment ($)", value=12500.0)
    v_age = st.number_input("Applicant Age", 18, 90, 35)
    v_days_emp = st.number_input("Days Employed (Negative)", value=-1500)

with st.sidebar.expander("🌐 External Risk Scores (EXT)", expanded=True):
    ext_1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5)
    ext_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.6)
    ext_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.4)

with st.sidebar.expander("📝 History Worksheet (Auto-Calc)", expanded=False):
    r_due = st.number_input("Total Amount Due ($)", value=10000.0)
    r_paid = st.number_input("Total Amount Actually Paid ($)", value=9800.0)
    c_pay_ratio = r_paid / r_due if r_due > 0 else 1.0
    r_late = st.number_input("Late Payment Count", 0, 500, 2)
    r_t_inst = st.number_input("Total Installments", 1, 500, 24)
    c_late_ratio = r_late / r_t_inst
    v_avg_delay = st.number_input("Avg Delay (Days)", value=2.5)
    v_prev_loans = st.number_input("Total Previous Loans", 0, 50, 3)
    v_active_loans = st.number_input("Active Loans", 0, 50, 1)
    c_act_ratio = v_active_loans / v_prev_loans if v_prev_loans > 0 else 0
    v_bureau_debt = st.slider("Bureau Debt-to-Credit Ratio", 0.0, 1.0, 0.3)

# ==========================================
# 6. EXECUTION ENGINE
# ==========================================
if st.button("🚀 EXECUTE RISK ASSESSMENT", type="primary"):
    
    # 1. Feature Engineering
    e_mean = (ext_1 + ext_2 + ext_3) / 3
    e_prod = ext_1 * ext_2 * ext_3
    c_i_ratio = v_credit / v_income if v_income > 0 else 0
    a_i_ratio = v_annuity / v_income if v_income > 0 else 0
    
    # 2. Build Data Dictionary
    raw_data = {
        'EXT_SOURCE_1': ext_1, 'EXT_SOURCE_2': ext_2, 'EXT_SOURCE_3': ext_3,
        'EXT_SOURCE_MEAN': e_mean, 'EXT_SOURCE_PRODUCT': e_prod,
        'AGE': v_age, 'DAYS_EMPLOYED': v_days_emp, 'AMT_INCOME_TOTAL': v_income,
        'AMT_CREDIT': np.log1p(v_credit), 'AMT_ANNUITY': v_annuity,
        'payment_ratio': c_pay_ratio, 'late_ratio': c_late_ratio,
        'avg_delay': v_avg_delay, 'credit_to_income_ratio': c_i_ratio,
        'annuity_to_income_ratio': a_i_ratio, 'active_loan_ratio': c_act_ratio,
        'total_previous_loans': v_prev_loans, 'active_loans': v_active_loans,
        'bureau_debt_to_credit_ratio': v_bureau_debt
    }

    # 3. Column Alignment
    if hasattr(model, 'feature_names_in_'): m_feats = model.feature_names_in_
    elif hasattr(model, 'feature_name_'): m_feats = model.feature_name_
    else: m_feats = model.booster_.feature_name()

    input_df = pd.DataFrame({f: [raw_data.get(f, 0.0)] for f in m_feats})

    # 4. Predict
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### SYSTEM VERDICT: <span style='color:{'#ff4b4b' if decision=='REJECTED' else '#28a745'}'>{decision}</span>", unsafe_allow_html=True)
    with col2:
        st.metric("Risk Probability", f"{prob:.2%}")

    # 5. SHAP Analysis
    with st.spinner("🤖 Analyzing risk drivers..."):
        explainer = shap.TreeExplainer(model)
        s_vals = explainer.shap_values(input_df)
        s_vals_c1 = s_vals[1][0] if isinstance(s_vals, list) else s_vals[0]
        
        t_idx = np.argsort(np.abs(s_vals_c1))[-5:][::-1]
        s_evidence = ""
        for i in t_idx:
            fn = input_df.columns[i]
            imp = s_vals_c1[i]
            direction = "INCREASED RISK" if imp > 0 else "DECREASED RISK"
            d = OFFICIAL_DICTIONARY.get(fn, fn)
            s_evidence += f"- {d} ({fn}): {imp:+.4f} ({direction})\n"

    # 6. LLM Generation
    with st.spinner("📝 Generating Underwriting Reports..."):
        report = generate_ai_feedback(prob, s_evidence, decision)
        st.markdown(report)
