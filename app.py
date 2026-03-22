import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from groq import Groq
from datetime import datetime

# ==========================================
# 1. SETUP & SECURITY
# ==========================================
st.set_page_config(page_title="Nexus AI Underwriting", page_icon="🛡️", layout="wide")

# Initialize Groq (The most stable free-tier LLM)
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("❌ GROQ_API_KEY is missing from Streamlit Secrets!")

# ==========================================
# 2. DATA DICTIONARY (For AI Knowledge)
# ==========================================
OFFICIAL_DICTIONARY = {
    "EXT_SOURCE_1": "Bureau Score 1 (Telco/Utilities)",
    "EXT_SOURCE_2": "Bureau Score 2 (Banking/Cashflow)",
    "EXT_SOURCE_3": "Bureau Score 3 (Digital Behavior)",
    "EXT_SOURCE_MEAN": "Aggregated Average Credit Score",
    "EXT_SOURCE_PRODUCT": "Credit Reliability Index",
    "AMT_CREDIT": "Total Loan Amount requested",
    "AMT_ANNUITY": "Monthly Payment amount",
    "AMT_INCOME_TOTAL": "Annual Income",
    "AGE": "Applicant Age",
    "DAYS_EMPLOYED": "Employment History Duration",
    "late_ratio": "Historical Payment Delay Rate",
    "payment_ratio": "Repayment Consistency (Paid vs Due)",
    "avg_delay": "Average Days Late",
    "credit_to_income_ratio": "Debt-to-Income Ratio",
    "active_loan_ratio": "Active Debt Proportion"
}

# ==========================================
# 3. ROBUST LLM ENGINE (Llama 3.3)
# ==========================================
def generate_ai_report(probability, shap_evidence, decision, client_name):
    # Capture current date
    today = datetime.now().strftime("%B %d, %Y")
    
    # Forceful prompt to prevent placeholders
    prompt = f"""
    You are the Senior Underwriter at Nexus Bank. Write a FINAL credit report.
    
    DATA FOR REPORT:
    Today's Date: {today}
    Applicant Name: {client_name}
    Final Decision: {decision}
    Risk Probability: {probability:.2%}
    
    KEY DRIVERS (SHAP Analysis):
    {shap_evidence}
    
    INSTRUCTIONS:
    1. Provide a 'TECHNICAL MEMO' for the bank committee.
    2. Provide a 'CLIENT LETTER' addressed to {client_name}.
    3. MANDATORY: Do NOT use brackets like [Date] or [Client Name]. Use the real data provided above.
    4. SIMPLIFY: Keep it clear and professional. Translate tech terms (e.g., 'late_ratio' to 'payment history').
    5. MATH: A positive (+) SHAP means Increased Risk. A negative (-) SHAP means a Strength.
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional bank official. You never use placeholders. You always use the specific names and dates provided."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1 # Keep it strictly factual
    )
    return chat_completion.choices[0].message.content

# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI: SIDEBAR INPUTS
# ==========================================
st.title("🛡️ Nexus AI: Cognitive Credit Underwriting Suite")
st.markdown("#### *Precision Risk Decisioning & Explainable Reports*")
st.divider()

st.sidebar.header("👤 Applicant Details")
c_name = st.sidebar.text_input("Full Client Name", "Vinay Kumar") # Added as requested

st.sidebar.header("📊 Application Dossier")

with st.sidebar.expander("💰 Financials", expanded=True):
    v_inc = st.number_input("Annual Income ($)", value=60000.0)
    v_cred = st.number_input("Requested Loan ($)", value=250000.0)
    v_ann = st.number_input("Monthly Installment ($)", value=12500.0)
    v_age = st.number_input("Age (Years)", 18, 90, 35)
    v_days = st.number_input("Days Employed (Negative)", value=-1500)

with st.sidebar.expander("🌐 External Sources", expanded=True):
    ext_1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5)
    ext_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.6)
    ext_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.4)

with st.sidebar.expander("📝 History Worksheet", expanded=False):
    r_due = st.number_input("Total Due", value=10000.0)
    r_paid = st.number_input("Total Paid", value=9800.0)
    c_p_ratio = r_paid / r_due if r_due > 0 else 1.0
    r_late = st.number_input("Late Count", 0, 500, 2)
    r_inst = st.number_input("Total Inst.", 1, 500, 24)
    c_l_ratio = r_late / r_inst
    v_delay = st.number_input("Avg Delay", value=2.5)
    v_prev = st.number_input("Prev Loans", 0, 50, 3)
    v_act = st.number_input("Active Loans", 0, 50, 1)
    c_act_r = v_act / v_prev if v_prev > 0 else 0
    v_b_debt = st.slider("Bureau Debt Ratio", 0.0, 1.0, 0.3)

# ==========================================
# 6. EVALUATION ENGINE
# ==========================================
if st.button("🚀 EXECUTE RISK ASSESSMENT", type="primary"):
    
    # Feature Engineering (Calculated internally so employee doesn't have to)
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

    # Column Alignment logic
    m_feats = model.feature_name_ if hasattr(model, 'feature_name_') else model.booster_.feature_name()
    input_df = pd.DataFrame({f: [raw_data.get(f, 0.0)] for f in m_feats})

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    st.divider()
    res1, res2 = st.columns(2)
    with res1:
        st.markdown(f"### VERDICT: <span style='color:{'red' if decision=='REJECTED' else 'green'}'>{decision}</span>", unsafe_allow_html=True)
    with res2:
        st.metric("Risk Probability", f"{prob:.2%}")

    # SHAP Explainability
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

    # LLM Report Generation
    with st.spinner("📝 Generating Final Underwriting Report..."):
        try:
            report = generate_ai_report(prob, s_evidence, decision, c_name)
            st.markdown(report)
        except Exception as e:
            st.error(f"Groq API Error: {str(e)}")
