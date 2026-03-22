import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from groq import Groq

# ==========================================
# 1. SETUP & SECURITY
# ==========================================
st.set_page_config(page_title="Nexus AI Underwriting", page_icon="🛡️", layout="wide")

# Initialize Groq Client
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("❌ GROQ_API_KEY is missing from Streamlit Secrets!")

# ==========================================
# 2. DATA DICTIONARY (For Tooltips & AI)
# ==========================================
OFFICIAL_DICTIONARY = {
    "EXT_SOURCE_1": "Alt Bureau Score 1 (Telco/Utilities)",
    "EXT_SOURCE_2": "Alt Bureau Score 2 (Cash Flow/Banking)",
    "EXT_SOURCE_3": "Alt Bureau Score 3 (Digital Behavior)",
    "EXT_SOURCE_MEAN": "Aggregated Average External Score",
    "EXT_SOURCE_PRODUCT": "External Source Reliability Index",
    "AMT_CREDIT": "Loan Amount (Log Transformed)",
    "AMT_ANNUITY": "Monthly Installment Amount",
    "AMT_INCOME_TOTAL": "Total Annual Income",
    "AGE": "Applicant Age (Years)",
    "DAYS_EMPLOYED": "Employment Duration (Days)",
    "late_ratio": "Historical Installment Delay Rate",
    "payment_ratio": "Repayment Consistency (Paid vs Due)",
    "avg_delay": "Average Days Late per Payment",
    "credit_to_income_ratio": "Total Debt-to-Income Ratio",
    "annuity_to_income_ratio": "Monthly Payment-to-Income Ratio",
    "active_loan_ratio": "Proportion of Active vs. Total Loans"
}

# ==========================================
# 3. LLM ENGINE (Llama 3.3)
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision):
    prompt = f"""
    You are a Senior Credit Underwriter. Analyze this specific model output:
    Verdict: {decision}
    Risk Probability: {probability:.2%}
    
    Top Mathematical Risk Drivers (SHAP Values):
    {shap_evidence}
    
    TASK:
    1. Write an INTERNAL BANK MEMO: Technical and data-driven analysis.
    2. Write a PROFESSIONAL CLIENT LETTER: Empathetic and clear (No technical jargon).
    Note: Positive impact (+) increases risk; Negative impact (-) decreases risk.
    """
    
    # Updated to the new Llama 3.3 model name
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert bank underwriter."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2
    )
    return chat_completion.choices[0].message.content

# ==========================================
# 4. LOAD MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI LAYOUT & SIDEBAR
# ==========================================
st.title("🛡️ Nexus AI: Cognitive Credit Underwriting Suite")
st.markdown("#### *Explainable AI (XAI) for Risk Decisioning*")
st.divider()

st.sidebar.header("📊 Application Dossier")

with st.sidebar.expander("💰 Financials & Bio", expanded=True):
    v_inc = st.number_input("Annual Income ($)", value=60000.0, help=OFFICIAL_DICTIONARY["AMT_INCOME_TOTAL"])
    v_cred = st.number_input("Requested Loan ($)", value=250000.0, help=OFFICIAL_DICTIONARY["AMT_CREDIT"])
    v_ann = st.number_input("Monthly Installment ($)", value=12500.0, help=OFFICIAL_DICTIONARY["AMT_ANNUITY"])
    v_age = st.number_input("Age (Years)", 18, 90, 35, help=OFFICIAL_DICTIONARY["AGE"])
    v_days = st.number_input("Days Employed (Negative)", value=-1500, help=OFFICIAL_DICTIONARY["DAYS_EMPLOYED"])

with st.sidebar.expander("🌐 External Risk (EXT)", expanded=True):
    ext_1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, help=OFFICIAL_DICTIONARY["EXT_SOURCE_1"])
    ext_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.6, help=OFFICIAL_DICTIONARY["EXT_SOURCE_2"])
    ext_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.4, help=OFFICIAL_DICTIONARY["EXT_SOURCE_3"])

with st.sidebar.expander("📝 History Worksheet (Auto-Calc)", expanded=False):
    st.caption("AI calculates ratios from raw historical data.")
    r_due = st.number_input("Total Amount Due ($)", value=10000.0)
    r_paid = st.number_input("Total Amount Paid ($)", value=9800.0)
    c_p_ratio = r_paid / r_due if r_due > 0 else 1.0
    
    r_late = st.number_input("Late Payment Count", 0, 500, 2)
    r_t_inst = st.number_input("Total Installments History", 1, 500, 24)
    c_l_ratio = r_late / r_t_inst
    
    v_delay = st.number_input("Avg Delay (Days)", value=2.5, help=OFFICIAL_DICTIONARY["avg_delay"])
    v_prev = st.number_input("Total Prev Loans", 0, 50, 3)
    v_act = st.number_input("Active Loans", 0, 50, 1)
    c_act_r = v_act / v_prev if v_prev > 0 else 0
    v_b_debt = st.slider("Bureau Debt Ratio", 0.0, 1.0, 0.3)

# ==========================================
# 6. EXECUTION ENGINE
# ==========================================
if st.button("🚀 EXECUTE RISK ASSESSMENT", type="primary"):
    
    # 1. Feature Engineering (Phase 3 & 4)
    raw_data = {
        'EXT_SOURCE_1': ext_1, 'EXT_SOURCE_2': ext_2, 'EXT_SOURCE_3': ext_3,
        'EXT_SOURCE_MEAN': (ext_1 + ext_2 + ext_3) / 3,
        'EXT_SOURCE_PRODUCT': ext_1 * ext_2 * ext_3,
        'AGE': v_age, 'DAYS_EMPLOYED': v_days, 'AMT_INCOME_TOTAL': v_inc,
        'AMT_CREDIT': np.log1p(v_cred), 'AMT_ANNUITY': v_ann,
        'payment_ratio': c_p_ratio, 'late_ratio': c_l_ratio, 'avg_delay': v_delay,
        'credit_to_income_ratio': v_cred / v_inc if v_inc > 0 else 0,
        'annuity_to_income_ratio': v_ann / v_inc if v_inc > 0 else 0,
        'active_loan_ratio': c_act_r, 'total_previous_loans': v_prev,
        'active_loans': v_act, 'bureau_debt_to_credit_ratio': v_b_debt
    }

    # 2. Column Alignment logic
    m_feats = model.feature_name_ if hasattr(model, 'feature_name_') else model.booster_.feature_name()
    input_df = pd.DataFrame({f: [raw_data.get(f, 0.0)] for f in m_feats})

    # 3. Model Prediction
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    st.divider()
    res1, res2 = st.columns(2)
    with res1:
        st.markdown(f"### VERDICT: <span style='color:{'red' if decision=='REJECTED' else 'green'}'>{decision}</span>", unsafe_allow_html=True)
    with res2:
        st.metric("Risk Probability", f"{prob:.2%}")

    # 4. SHAP Explainability (Inside Button Scope)
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

    # 5. LLM Report Generation (Groq / Llama 3.3)
    with st.spinner("📝 Generating Final Underwriting Report..."):
        try:
            report = generate_ai_feedback(prob, s_evidence, decision)
            st.markdown(report)
        except Exception as e:
            st.error(f"Groq API Error: {str(e)}")
