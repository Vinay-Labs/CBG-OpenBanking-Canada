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
# 3. ROBUST AI GENERATION ENGINE
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision, applicant_name="Applicant"):
    # The "Master Prompt" designed for Banking Compliance
    prompt = f"""
    You are a Senior Credit Risk Underwriter and Compliance Officer. 
    A loan application has been processed by our LightGBM Gradient Boosting model.
    
    VERDICT: {decision}
    DEFAULT RISK PROBABILITY: {probability:.2%}
    
    SHAP EXPLAINABILITY DATA (The "Why"):
    {shap_evidence}
    
    INSTRUCTIONS:
    Provide a response with two distinct sections using Markdown headers.

    ### 🔒 INTERNAL UNDERWRITING MEMO (For Bank Staff)
    - Explain the model's decision using technical terms (e.g., "High variance in EXT_SOURCE_3" or "Log-transformed Credit Amount").
    - Refer specifically to the SHAP values. Explain how the combination of features pushed the probability to {probability:.2%}.
    - Be objective and data-driven. Mention if the decision was "borderline" (near the 25% threshold).

    ### ✉️ ADVERSE ACTION / APPROVAL LETTER (For {applicant_name})
    - Use professional, empathetic banking language.
    - DO NOT use technical terms like 'SHAP', 'EXT_SOURCE', 'Log-transform', or 'Ratio'.
    - TRANSLATE the data: 
        - If EXT_SOURCE_MEAN is a risk: "Alternative credit markers indicate lower financial stability."
        - If late_ratio is a risk: "Recent history of installment delays."
        - If debt_to_income is a risk: "High debt obligations relative to reported income."
    - If REJECTED: Clearly state the top 2 reasons for denial as per regulatory requirements.
    - If APPROVED: Welcome them and mention the strengths of their application.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional bank underwriter. You translate machine learning SHAP values into clear business logic."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 # Low temperature for consistency and accuracy
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM Bridge Error: {str(e)}\n\n**Manual Fallback Summary:** The model {decision} the application with a risk score of {probability:.2%} based on the provided financial metrics."

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
# 6. NEXUS AI EVALUATION ENGINE (FINAL VERSION)
# ==========================================
if st.button("🚀 EXECUTE RISK ASSESSMENT", type="primary"):
    
    # 1. Background Feature Engineering (Automating the SQL/Phase 3 logic)
    ext_mean = (ext_1 + ext_2 + ext_3) / 3
    ext_prod = ext_1 * ext_2 * ext_3
    c_to_i = credit_amt / income if income > 0 else 0
    a_to_i = annuity / income if income > 0 else 0
    
    # Map all inputs to the model's required format
    raw_data = {
        'EXT_SOURCE_1': ext_1, 'EXT_SOURCE_2': ext_2, 'EXT_SOURCE_3': ext_3,
        'EXT_SOURCE_MEAN': ext_mean, 'EXT_SOURCE_PRODUCT': ext_prod,
        'AGE': age, 'AMT_INCOME_TOTAL': income, 'AMT_CREDIT': np.log1p(credit_amt),
        'AMT_ANNUITY': annuity, 'payment_ratio': payment_ratio,
        'late_ratio': late_ratio, 'avg_delay': avg_delay,
        'credit_to_income_ratio': c_to_i, 'annuity_to_income_ratio': a_to_i
    }

    # Robust Feature Alignment (Checks all possible model attribute names)
    if hasattr(model, 'feature_names_in_'): 
        model_features = model.feature_names_in_
    elif hasattr(model, 'feature_name_'):
        model_features = model.feature_name_
    else:
        model_features = model.booster_.feature_name()

    # Create 1-row DataFrame horizontally (The fix for the Shape Error)
    processed_input = {feat: [raw_data.get(feat, 0.0)] for feat in model_features}
    input_df = pd.DataFrame(processed_input)

    # 2. Model Prediction
    prob = model.predict_proba(input_df)[0][1]
    # Business threshold of 25% (Adjust this if your notebook uses a different one)
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    # UI Results Display
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        color = "#ff4b4b" if decision == "REJECTED" else "#28a745"
        st.markdown(f"### SYSTEM VERDICT: <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)
    with res_col2:
        st.metric("Risk Probability", f"{prob:.2%}")

    # 3. SHAP Explainability Logic (The "Why")
    with st.spinner("🤖 Nexus AI is analyzing risk drivers..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # Handle LightGBM/SHAP version differences for Class 1 (Default)
        if isinstance(shap_values, list):
            individual_shap = shap_values[1][0] 
        else:
            individual_shap = shap_values[0]

        # Get top 5 features by absolute impact
        top_indices = np.argsort(np.abs(individual_shap))[-5:][::-1]
        
        shap_evidence = ""
        for i in top_indices:
            f_name = input_df.columns[i]
            val_impact = individual_shap[i]
            direction = "INCREASED RISK" if val_impact > 0 else "DECREASED RISK"
            friendly_name = OFFICIAL_DICTIONARY.get(f_name, f_name)
            
            # This string is what the LLM reads to understand the math
            shap_evidence += f"- {friendly_name} ({f_name}): {val_impact:+.4f} ({direction})\n"

    # 4. Generative AI Report (Final Presentation Piece)
    with st.spinner("📝 Generating Underwriting Reports..."):
        # This function must be defined at the top of your app.py
        ai_report = generate_ai_feedback(prob, shap_evidence, decision)
        st.markdown(ai_report)
