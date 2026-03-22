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

try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai.api_key = "YOUR_API_KEY_HERE"

# ==========================================
# 2. DATA DICTIONARY (For LLM & Tooltips)
# ==========================================
OFFICIAL_DICTIONARY = {
    "EXT_SOURCE_1": "External Source 1: Credit Bureau/Telecom Score",
    "EXT_SOURCE_2": "External Source 2: Transactional Cash Flow Score",
    "EXT_SOURCE_3": "External Source 3: Alternative Behavioral Score",
    "late_ratio": "Percentage of past installments that were paid after the due date.",
    "payment_ratio": "Calculated as (Total Paid / Total Due). Values < 1 indicate underpayment.",
    "avg_delay": "Average days late across all installments. Positive is late, negative is early.",
    "credit_to_income_ratio": "Proportion of requested loan amount relative to annual income.",
    "active_loan_ratio": "Proportion of total historical loans that are currently active."
}

# ==========================================
# 3. AI GENERATION ENGINE
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision):
    prompt = f"""
    You are a Senior Risk Underwriter at a global bank. 
    Review Decision: {decision} | Model-Calculated Risk: {probability:.2%}
    
    SHAP EXPLAINABILITY DATA:
    {shap_evidence}
    
    INSTRUCTIONS:
    1. Write an INTERNAL RISK MEMO: Use technical terms (like EXT_SOURCE_PRODUCT). Explain the logic.
    2. Write a CLIENT DECISION LETTER: Use empathetic, professional language. 
       - If rejected, use the data to explain the specific risk (e.g., 'recent payment delays' instead of 'late_ratio').
       - If approved, mention their 'strong financial stability indicators'.
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
        return f"AI Generation Offline. Logic Summary: {decision} with {probability:.2%} risk."

# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI LAYOUT: NEXUS AI SUITE
# ==========================================
st.title("🛡️ Nexus AI: Cognitive Credit Underwriting Suite")
st.markdown("#### *High-Precision Default Risk Prediction & Explainable AI (XAI)*")
st.divider()

# SIDEBAR INPUTS
st.sidebar.header("📊 Application Dossier")

# --- SECTION 1: FINANCIALS ---
with st.sidebar.expander("💰 Core Financials", expanded=True):
    income = st.number_input("Annual Income ($)", value=60000.0, help="Total gross income of the applicant.")
    credit_amt = st.number_input("Requested Loan ($)", value=250000.0, help="The total amount the client wants to borrow.")
    annuity = st.number_input("Monthly Installment ($)", value=12500.0, help="The amount the client will pay every month.")
    age = st.number_input("Applicant Age", 18, 90, 35, help="Standard age in years.")

# --- SECTION 2: EXT SOURCES ---
with st.sidebar.expander("🌐 External Risk Scores", expanded=True):
    ext_1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, help="Score from External Credit Bureau 1.")
    ext_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.6, help="Internal Score based on Cash Flow/Banking history.")
    ext_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.4, help="Score from Alternative Digital Behavioral Data.")

# --- SECTION 3: CALCULATION WORKSHEET FOR EMPLOYEE ---
with st.sidebar.expander("📝 History Worksheet (Auto-Calc)", expanded=False):
    st.caption("Enter raw history; the AI calculates the ratios below.")
    tot_due = st.number_input("Total Amount Ever Due ($)", value=10000.0)
    tot_paid = st.number_input("Total Amount Actually Paid ($)", value=9800.0)
    
    # Internal Calculation:
    payment_ratio = tot_paid / tot_due if tot_due > 0 else 1.0
    st.write(f"**Calculated Payment Ratio:** {payment_ratio:.2f}")

    late_count = st.number_input("Number of Late Payments", 0, 500, 2)
    total_inst = st.number_input("Total Installment Count", 1, 500, 24)
    
    # Internal Calculation:
    late_ratio = late_count / total_inst
    st.write(f"**Calculated Late Ratio:** {late_ratio:.2%}")

    avg_delay = st.number_input("Avg Delay (Days)", value=2.0, help="Decimal allowed: e.g., 2.5 days means across all loans, the average lateness was 2.5 days.")

# ==========================================
# 6. EVALUATION ENGINE
# ==========================================
if st.button("🚀 EXECUTE RISK ASSESSMENT", type="primary"):
    
    # 1. Feature Engineering
    ext_mean = (ext_1 + ext_2 + ext_3) / 3
    ext_prod = ext_1 * ext_2 * ext_3
    c_to_i = credit_amt / income if income > 0 else 0
    a_to_i = annuity / income if income > 0 else 0
    
    # 2. Alignment with Model Features
    # This block uses a dictionary comprehension to ensure a 1-row DataFrame is built perfectly
    raw_data = {
        'EXT_SOURCE_1': ext_1, 'EXT_SOURCE_2': ext_2, 'EXT_SOURCE_3': ext_3,
        'EXT_SOURCE_MEAN': ext_mean, 'EXT_SOURCE_PRODUCT': ext_prod,
        'AGE': age, 'AMT_INCOME_TOTAL': income, 'AMT_CREDIT': np.log1p(credit_amt),
        'AMT_ANNUITY': annuity, 'payment_ratio': payment_ratio,
        'late_ratio': late_ratio, 'avg_delay': avg_delay,
        'credit_to_income_ratio': c_to_i, 'annuity_to_income_ratio': a_to_i
    }

    # FIX: Robustly get feature names and force the shape
    if hasattr(model, 'feature_names_in_'): model_features = model.feature_names_in_
    else: model_features = model.feature_name_

    # Create DataFrame: Ensure it is 1 Row, N Columns
    # Passing as [list] inside dictionary ensures horizontal orientation
    processed_input = {feat: [raw_data.get(feat, 0.0)] for feat in model_features}
    input_df = pd.DataFrame(processed_input)

    # 3. Predict Decision
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    # UI Display
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        color = "red" if decision == "REJECTED" else "green"
        st.markdown(f"### SYSTEM VERDICT: <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)
    with res_col2:
        st.metric("Risk Probability", f"{prob:.2%}")

    # 4. SHAP Logic
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    shap_vals_class1 = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    
    top_idx = np.argsort(np.abs(shap_vals_class1))[-5:]
    shap_evidence = ""
    for i in top_idx:
        f_name = input_df.columns[i]
        impact = shap_vals_class1[i]
        desc = OFFICIAL_DICTIONARY.get(f_name, "Internal Risk Proxy")
        shap_evidence += f"- {desc} ({f_name}): {impact:+.4f}\n"

    # 5. LLM Report
    with st.spinner("GenAI Analyzing Risk Drivers..."):
        full_report = generate_ai_feedback(prob, shap_evidence, decision)
        st.divider()
        st.markdown(full_report)
