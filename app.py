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

# Check for API Key in Streamlit Secrets
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ CRITICAL ERROR: 'OPENAI_API_KEY' not found in Streamlit Secrets. Go to Settings > Secrets and add it.")

# ==========================================
# 2. DATA DICTIONARY (Tooltips & Context)
# ==========================================
OFFICIAL_DICTIONARY = {
    "EXT_SOURCE_1": "External Source 1: Credit Bureau/Telecom history score.",
    "EXT_SOURCE_2": "External Source 2: Transactional Cash Flow & Banking consistency.",
    "EXT_SOURCE_3": "External Source 3: Alternative Digital Behavioral data.",
    "AMT_CREDIT": "The total amount of the loan requested.",
    "AMT_ANNUITY": "The monthly payment amount for the requested loan.",
    "AMT_INCOME_TOTAL": "The total gross annual income of the applicant.",
    "AGE": "Applicant's age in years.",
    "DAYS_EMPLOYED": "Total days of current employment (Negative value).",
    "total_previous_loans": "Total number of loans taken in the past.",
    "active_loans": "Number of previous loans currently still open.",
    "avg_delay": "Average days late across historical payments (Decimal allowed).",
    "late_ratio": "Percentage of past payments made after the due date.",
    "payment_ratio": "Ratio of (Amount Paid / Amount Due). 1.0 is ideal.",
    "credit_to_income_ratio": "Requested loan amount divided by annual income.",
    "annuity_to_income_ratio": "Monthly payment divided by annual income.",
    "active_loan_ratio": "Proportion of previous loans currently active."
}

# ==========================================
# 3. ROBUST AI GENERATION ENGINE
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision):
    prompt = f"""
    You are a Senior Risk Underwriter. Review this model output:
    Decision: {decision} | Default Risk: {probability:.2%}
    
    SHAP MATH EVIDENCE (Feature Impacts):
    {shap_evidence}
    
    TASK: Provide an 'Internal Underwriting Memo' and a 'Professional Letter' for the client.
    - Translate 'EXT_SOURCE' to 'Alternative Credit Indicators'.
    - Explain that positive (+) impact increases risk and negative (-) impact is a strength.
    """
    try:
        # Use the modern OpenAI 1.x syntax
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an expert bank underwriter."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        # IMPROVED ERROR REPORTING: This will show you EXACTLY why it failed
        return f"⚠️ LLM Offline. Reason: {str(e)}\n\n(Please check your OpenAI Billing/Quota or API Key in Streamlit Secrets)."

# ==========================================
# 4. LOAD MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    # Ensure this filename matches your GitHub exactly!
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI: NEXUS AI SUITE
# ==========================================
st.title("🛡️ Nexus AI: Cognitive Credit Underwriting Suite")
st.markdown("#### *High-Precision Default Risk Prediction & Explainable AI (XAI)*")
st.divider()

st.sidebar.header("📊 Application Dossier")

# --- SECTION 1: FINANCIALS ---
with st.sidebar.expander("💰 Core Financials", expanded=True):
    val_income = st.number_input("Annual Income ($)", value=60000.0, help=OFFICIAL_DICTIONARY["AMT_INCOME_TOTAL"])
    val_credit = st.number_input("Requested Loan ($)", value=250000.0, help=OFFICIAL_DICTIONARY["AMT_CREDIT"])
    val_annuity = st.number_input("Monthly Installment ($)", value=12500.0, help=OFFICIAL_DICTIONARY["AMT_ANNUITY"])
    val_age = st.number_input("Applicant Age", 18, 90, 35, help=OFFICIAL_DICTIONARY["AGE"])
    val_days_emp = st.number_input("Days Employed (Negative)", value=-1500, help=OFFICIAL_DICTIONARY["DAYS_EMPLOYED"])

# --- SECTION 2: EXT SOURCES ---
with st.sidebar.expander("🌐 External Risk Scores (EXT)", expanded=True):
    ext_1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, help=OFFICIAL_DICTIONARY["EXT_SOURCE_1"])
    ext_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.6, help=OFFICIAL_DICTIONARY["EXT_SOURCE_2"])
    ext_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.4, help=OFFICIAL_DICTIONARY["EXT_SOURCE_3"])

# --- SECTION 3: HISTORY WORKSHEET ---
with st.sidebar.expander("📝 History Worksheet (Auto-Calc)", expanded=False):
    st.caption("Enter raw data; the AI calculates the ratios.")
    raw_due = st.number_input("Total Amount Due ($)", value=10000.0)
    raw_paid = st.number_input("Total Amount Actually Paid ($)", value=9800.0)
    calc_payment_ratio = raw_paid / raw_due if raw_due > 0 else 1.0
    
    raw_late = st.number_input("Late Payment Count", 0, 500, 2)
    raw_total_inst = st.number_input("Total Installments", 1, 500, 24)
    calc_late_ratio = raw_late / raw_total_inst
    
    val_avg_delay = st.number_input("Avg Delay (Days)", value=2.5, help=OFFICIAL_DICTIONARY["avg_delay"])
    
    val_prev_loans = st.number_input("Total Previous Loans", 0, 50, 3)
    val_active_loans = st.number_input("Current Active Loans", 0, 50, 1)
    calc_active_ratio = val_active_loans / val_prev_loans if val_prev_loans > 0 else 0
    val_bureau_debt = st.slider("Bureau Debt-to-Credit (%)", 0.0, 1.0, 0.3)

# ==========================================
# 6. ENGINE EXECUTION
# ==========================================
if st.button("🚀 EXECUTE RISK ASSESSMENT", type="primary"):
    
    # 1. Feature Engineering
    ext_mean = (ext_1 + ext_2 + ext_3) / 3
    ext_prod = ext_1 * ext_2 * ext_3
    c_to_i = val_credit / val_income if val_income > 0 else 0
    a_to_i = val_annuity / val_income if val_income > 0 else 0
    
    # 2. Build the Raw Data Dictionary
    raw_data = {
        'EXT_SOURCE_1': ext_1, 'EXT_SOURCE_2': ext_2, 'EXT_SOURCE_3': ext_3,
        'EXT_SOURCE_MEAN': ext_mean, 'EXT_SOURCE_PRODUCT': ext_prod,
        'AGE': val_age, 'DAYS_EMPLOYED': val_days_emp, 'AMT_INCOME_TOTAL': val_income,
        'AMT_CREDIT': np.log1p(val_credit), 'AMT_ANNUITY': val_annuity,
        'payment_ratio': calc_payment_ratio, 'late_ratio': calc_late_ratio,
        'avg_delay': val_avg_delay, 'credit_to_income_ratio': c_to_i,
        'annuity_to_income_ratio': a_to_i, 'active_loan_ratio': calc_active_ratio,
        'total_previous_loans': val_prev_loans, 'active_loans': val_active_loans,
        'bureau_debt_to_credit_ratio': val_bureau_debt
    }

    # 3. Match Columns exactly to Model
    if hasattr(model, 'feature_names_in_'): model_features = model.feature_names_in_
    elif hasattr(model, 'feature_name_'): model_features = model.feature_name_
    else: model_features = model.booster_.feature_name()

    # Create 1-row DataFrame (Correcting Shape Errors)
    input_df = pd.DataFrame({feat: [raw_data.get(feat, 0.0)] for feat in model_features})

    # 4. Predict
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    st.divider()
    res1, res2 = st.columns(2)
    with res1:
        st.markdown(f"### SYSTEM VERDICT: <span style='color:{'#ff4b4b' if decision=='REJECTED' else '#28a745'}'>{decision}</span>", unsafe_allow_html=True)
    with res2:
        st.metric("Risk Probability", f"{prob:.2%}")

    # 5. SHAP & AI (Inside button scope)
    with st.spinner("🤖 Nexus AI Analyzing Drivers..."):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_df)
        shap_vals_c1 = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        
        top_idx = np.argsort(np.abs(shap_vals_c1))[-5:][::-1]
        shap_evidence = ""
        for i in top_idx:
            f_name = input_df.columns[i]
            impact = shap_vals_c1[i]
            dir_text = "INCREASED RISK" if impact > 0 else "DECREASED RISK"
            desc = OFFICIAL_DICTIONARY.get(f_name, f_name)
            shap_evidence += f"- {desc} ({f_name}): {impact:+.4f} ({dir_text})\n"

    # 6. AI Generation
    with st.spinner("📝 Generating Underwriting Reports..."):
        report = generate_ai_feedback(prob, shap_evidence, decision)
        st.markdown(report)
