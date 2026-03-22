import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import openai

# ==========================================
# 1. PAGE CONFIG & SECRETS
# ==========================================
st.set_page_config(page_title="AI Underwriting Engine", page_icon="🏦", layout="wide")

try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai.api_key = "YOUR_FALLBACK_API_KEY_HERE" # DO NOT hardcode your real key in GitHub

# ==========================================
# 2. OFFICIAL DATA DICTIONARY (Updated with SQL Features)
# ==========================================
OFFICIAL_DICTIONARY = {
    # Engineered Ratios (from your SQL feature_table)
    "credit_to_income_ratio": "Credit-to-Income Ratio (Requested Loan vs Total Income)",
    "annuity_to_income_ratio": "Annuity-to-Income Ratio (Monthly Payment vs Total Income)",
    "active_loan_ratio": "Active Loan Ratio (Proportion of previous loans still active)",
    
    # Bureau Aggregations (from your SQL bureau_agg)
    "total_previous_loans": "Total Number of Previous Loans on File",
    "avg_credit_sum": "Average Amount of Previous Loans",
    "bureau_debt_to_credit_ratio": "Historical Debt-to-Credit Ratio across previous loans",
    
    # Installment Aggregations (from your SQL installment_agg)
    "total_installments": "Total Number of Past Installment Payments",
    "avg_delay": "Average Payment Delay (Positive = Late, Negative = Early)",
    "late_ratio": "Late Payment Ratio (Percentage of installments paid past the due date)",
    "payment_ratio": "Payment Ratio (Actual amount paid vs Prescribed installment amount)",
    
    # Alternative/Phase 3 Features
    "EXT_SOURCE_1": "Alternative Data Proxy 1 (Telco/Utility)",
    "EXT_SOURCE_2": "Alternative Data Proxy 2 (Cash Flow)",
    "EXT_SOURCE_3": "Alternative Data Proxy 3 (Behavioral)",
    "EXT_SOURCE_MEAN": "Aggregated Alternative Risk Score",
    "EXT_SOURCE_PRODUCT": "Combined Alternative Data Reliability Index",
    
    # Core Demographics (Transformed)
    "AGE": "Applicant Age (Years)",
    "DAYS_EMPLOYED": "Length of Current Employment (Days)",
    "AMT_CREDIT": "Total Requested Credit Amount (Log Transformed)",
    "AMT_INCOME_TOTAL": "Total Applicant Income"
}

# ==========================================
# 3. AI GENERATION ENGINE
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision):
    prompt = f"""
    You are an expert AI Underwriting Assistant for a FinTech bank.
    
    Applicant Decision: {decision} (Probability of Default: {probability * 100:.1f}%)
    
    EVIDENCE (Calculated via SHAP values):
    - A positive (+) impact means the feature INCREASED the risk of default.
    - A negative (-) impact means the feature DECREASED the risk of default.
    
    {shap_evidence}
    
    ### INTERNAL BANK MEMO
    Write a brief, technical assessment for the loan officer. Use the raw feature names. Explain how the historical installment behavior and alternative data drove this decision based on the SHAP math.
    
    ### APPLICANT LETTER
    Write a polite, regulatory-compliant letter to the applicant explaining the decision. 
    - DO NOT use technical variable names (like 'EXT_SOURCE_2' or 'late_ratio'). Translate them using the descriptive names provided in the evidence.
    - DO NOT mention "SHAP", "Machine Learning", or exact probability numbers.
    - If rejected, gently explain the specific factors that increased their risk as required by the Equal Credit Opportunity Act.
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
        return f"⚠️ AI Generation Failed. Error: {str(e)}"

# ==========================================
# 4. CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI & RAW INPUTS (What the Loan Officer Types)
# ==========================================
st.title("🏦 FinTech AI: Thin-File Underwriter")
st.markdown("Automated Risk Assessment powered by LightGBM, Alternative Data, and SHAP Explainability.")
st.divider()

st.sidebar.header("📝 Application Data Entry")

st.sidebar.subheader("1. Applicant Core Details")
age_years = st.sidebar.number_input("Age (Years)", min_value=18, max_value=100, value=35)
days_employed = st.sidebar.number_input("Days Employed (Negative format)", value=-1500)
amt_income = st.sidebar.number_input("Total Annual Income ($)", value=60000.0)
amt_credit = st.sidebar.number_input("Requested Loan Amount ($)", value=250000.0)
amt_annuity = st.sidebar.number_input("Proposed Monthly Annuity ($)", value=12500.0)

st.sidebar.subheader("2. External Alt Data")
ext_1 = st.sidebar.slider("Alt Proxy 1", 0.0, 1.0, 0.50)
ext_2 = st.sidebar.slider("Alt Proxy 2", 0.0, 1.0, 0.65)
ext_3 = st.sidebar.slider("Alt Proxy 3", 0.0, 1.0, 0.55)

st.sidebar.subheader("3. Historical Bureau & Installments")
total_prev_loans = st.sidebar.number_input("Total Previous Loans", min_value=0, value=3)
active_loans = st.sidebar.number_input("Currently Active Loans", min_value=0, value=1)
bureau_debt_ratio = st.sidebar.slider("Bureau Debt-to-Credit Ratio", 0.0, 1.0, 0.3)

total_installments = st.sidebar.number_input("Total Past Installments", min_value=0, value=24)
late_ratio = st.sidebar.slider("Late Payment Ratio", 0.0, 1.0, 0.05)
avg_delay = st.sidebar.number_input("Avg Payment Delay (Days, + is late)", value=-2.5)
payment_ratio = st.sidebar.slider("Payment Ratio (Paid vs Required)", 0.0, 1.5, 1.0)

# ==========================================
# 6. INPUT VALIDATION (Safety Checks)
# ==========================================
errors = []
if amt_income <= 0: errors.append("Annual Income must be greater than $0.")
if active_loans > total_prev_loans: errors.append("Active loans cannot exceed total previous loans.")
if amt_annuity >= amt_credit: errors.append("Monthly annuity cannot be larger than the total loan.")

if errors:
    for e in errors: st.error(f"⚠️ {e}")
    st.warning("Please fix input errors to proceed.")
    st.stop()

# ==========================================
# 7. ENGINE: AUTO-CALCULATIONS & PREDICTION
# ==========================================
if st.button("🚀 Run AI Underwriting Engine", type="primary"):
    
    # A. BACKGROUND CALCULATIONS (Your SQL & Phase 3 logic happens invisibly here)
    ext_mean = (ext_1 + ext_2 + ext_3) / 3
    ext_product = ext_1 * ext_2 * ext_3
    
    credit_to_income = amt_credit / amt_income
    annuity_to_income = amt_annuity / amt_income
    active_loan_ratio = active_loans / total_prev_loans if total_prev_loans > 0 else 0
    
    # Skewness Transform
    amt_credit_log = np.log1p(amt_credit)

    # B. BUILD THE EXACT MODEL INPUT DICTIONARY
    input_dict = {
        'AGE': age_years,
        'DAYS_EMPLOYED': days_employed,
        'AMT_INCOME_TOTAL': amt_income,
        'AMT_CREDIT': amt_credit_log,
        'AMT_ANNUITY': amt_annuity,
        'EXT_SOURCE_1': ext_1,
        'EXT_SOURCE_2': ext_2,
        'EXT_SOURCE_3': ext_3,
        'EXT_SOURCE_MEAN': ext_mean,
        'EXT_SOURCE_PRODUCT': ext_product,
        'total_previous_loans': total_prev_loans,
        'active_loans': active_loans,
        'bureau_debt_to_credit_ratio': bureau_debt_ratio,
        'total_installments': total_installments,
        'late_ratio': late_ratio,
        'avg_delay': avg_delay,
        'payment_ratio': payment_ratio,
        'credit_to_income_ratio': credit_to_income,
        'annuity_to_income_ratio': annuity_to_income,
        'active_loan_ratio': active_loan_ratio
    }
    
    # Safely pad missing columns with 0.0 based on what LightGBM expects
    full_input = {feat: input_dict.get(feat, 0.0) for feat in model.feature_name_}
    input_df = pd.DataFrame([full_input])

    # C. PREDICT
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("System Decision", f"{'❌' if decision=='REJECTED' else '✅'} {decision}")
    with col2:
        st.metric("Probability of Default", f"{prob:.2%}")

    # D. EXTRACT SHAP & FEED LLM
    with st.spinner("Extracting Mathematical SHAP Evidence..."):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_df)
        shap_vals_default = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        
        # Get Top 5 most impactful features (positive or negative)
        top_indices = np.argsort(np.abs(shap_vals_default))[-5:]
        
        shap_evidence = ""
        for i in top_indices:
            feat_name = input_df.columns[i]
            impact = shap_vals_default[i]
            desc = OFFICIAL_DICTIONARY.get(feat_name, "Internal Risk Metric")
            shap_evidence += f"- {feat_name} | {desc} | Impact: {impact:+.3f}\n"

    with st.spinner("Generating Regulatory-Compliant Letters..."):
        ai_report = generate_ai_feedback(prob, shap_evidence, decision)
        
        st.divider()
        if "### APPLICANT LETTER" in ai_report:
            memo, letter = ai_report.split("### APPLICANT LETTER")
            
            tab1, tab2 = st.tabs(["🔒 Internal Loan Officer Memo", "✉️ Applicant Letter"])
            with tab1:
                st.info(memo.replace("### INTERNAL BANK MEMO", "").strip())
            with tab2:
                st.success(letter.strip())
        else:
            st.write(ai_report)import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import openai

# ==========================================
# 1. PAGE CONFIG & SECRETS
# ==========================================
st.set_page_config(page_title="AI Underwriting Engine", page_icon="🏦", layout="wide")

try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai.api_key = "YOUR_FALLBACK_API_KEY_HERE" # DO NOT hardcode your real key in GitHub

# ==========================================
# 2. OFFICIAL DATA DICTIONARY (Updated with SQL Features)
# ==========================================
OFFICIAL_DICTIONARY = {
    # Engineered Ratios (from your SQL feature_table)
    "credit_to_income_ratio": "Credit-to-Income Ratio (Requested Loan vs Total Income)",
    "annuity_to_income_ratio": "Annuity-to-Income Ratio (Monthly Payment vs Total Income)",
    "active_loan_ratio": "Active Loan Ratio (Proportion of previous loans still active)",
    
    # Bureau Aggregations (from your SQL bureau_agg)
    "total_previous_loans": "Total Number of Previous Loans on File",
    "avg_credit_sum": "Average Amount of Previous Loans",
    "bureau_debt_to_credit_ratio": "Historical Debt-to-Credit Ratio across previous loans",
    
    # Installment Aggregations (from your SQL installment_agg)
    "total_installments": "Total Number of Past Installment Payments",
    "avg_delay": "Average Payment Delay (Positive = Late, Negative = Early)",
    "late_ratio": "Late Payment Ratio (Percentage of installments paid past the due date)",
    "payment_ratio": "Payment Ratio (Actual amount paid vs Prescribed installment amount)",
    
    # Alternative/Phase 3 Features
    "EXT_SOURCE_1": "Alternative Data Proxy 1 (Telco/Utility)",
    "EXT_SOURCE_2": "Alternative Data Proxy 2 (Cash Flow)",
    "EXT_SOURCE_3": "Alternative Data Proxy 3 (Behavioral)",
    "EXT_SOURCE_MEAN": "Aggregated Alternative Risk Score",
    "EXT_SOURCE_PRODUCT": "Combined Alternative Data Reliability Index",
    
    # Core Demographics (Transformed)
    "AGE": "Applicant Age (Years)",
    "DAYS_EMPLOYED": "Length of Current Employment (Days)",
    "AMT_CREDIT": "Total Requested Credit Amount (Log Transformed)",
    "AMT_INCOME_TOTAL": "Total Applicant Income"
}

# ==========================================
# 3. AI GENERATION ENGINE
# ==========================================
def generate_ai_feedback(probability, shap_evidence, decision):
    prompt = f"""
    You are an expert AI Underwriting Assistant for a FinTech bank.
    
    Applicant Decision: {decision} (Probability of Default: {probability * 100:.1f}%)
    
    EVIDENCE (Calculated via SHAP values):
    - A positive (+) impact means the feature INCREASED the risk of default.
    - A negative (-) impact means the feature DECREASED the risk of default.
    
    {shap_evidence}
    
    ### INTERNAL BANK MEMO
    Write a brief, technical assessment for the loan officer. Use the raw feature names. Explain how the historical installment behavior and alternative data drove this decision based on the SHAP math.
    
    ### APPLICANT LETTER
    Write a polite, regulatory-compliant letter to the applicant explaining the decision. 
    - DO NOT use technical variable names (like 'EXT_SOURCE_2' or 'late_ratio'). Translate them using the descriptive names provided in the evidence.
    - DO NOT mention "SHAP", "Machine Learning", or exact probability numbers.
    - If rejected, gently explain the specific factors that increased their risk as required by the Equal Credit Opportunity Act.
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
        return f"⚠️ AI Generation Failed. Error: {str(e)}"

# ==========================================
# 4. CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("lgbm_credit_model.pkl")

model = load_model()

# ==========================================
# 5. UI & RAW INPUTS (What the Loan Officer Types)
# ==========================================
st.title("🏦 FinTech AI: Thin-File Underwriter")
st.markdown("Automated Risk Assessment powered by LightGBM, Alternative Data, and SHAP Explainability.")
st.divider()

st.sidebar.header("📝 Application Data Entry")

st.sidebar.subheader("1. Applicant Core Details")
age_years = st.sidebar.number_input("Age (Years)", min_value=18, max_value=100, value=35)
days_employed = st.sidebar.number_input("Days Employed (Negative format)", value=-1500)
amt_income = st.sidebar.number_input("Total Annual Income ($)", value=60000.0)
amt_credit = st.sidebar.number_input("Requested Loan Amount ($)", value=250000.0)
amt_annuity = st.sidebar.number_input("Proposed Monthly Annuity ($)", value=12500.0)

st.sidebar.subheader("2. External Alt Data")
ext_1 = st.sidebar.slider("Alt Proxy 1", 0.0, 1.0, 0.50)
ext_2 = st.sidebar.slider("Alt Proxy 2", 0.0, 1.0, 0.65)
ext_3 = st.sidebar.slider("Alt Proxy 3", 0.0, 1.0, 0.55)

st.sidebar.subheader("3. Historical Bureau & Installments")
total_prev_loans = st.sidebar.number_input("Total Previous Loans", min_value=0, value=3)
active_loans = st.sidebar.number_input("Currently Active Loans", min_value=0, value=1)
bureau_debt_ratio = st.sidebar.slider("Bureau Debt-to-Credit Ratio", 0.0, 1.0, 0.3)

total_installments = st.sidebar.number_input("Total Past Installments", min_value=0, value=24)
late_ratio = st.sidebar.slider("Late Payment Ratio", 0.0, 1.0, 0.05)
avg_delay = st.sidebar.number_input("Avg Payment Delay (Days, + is late)", value=-2.5)
payment_ratio = st.sidebar.slider("Payment Ratio (Paid vs Required)", 0.0, 1.5, 1.0)

# ==========================================
# 6. INPUT VALIDATION (Safety Checks)
# ==========================================
errors = []
if amt_income <= 0: errors.append("Annual Income must be greater than $0.")
if active_loans > total_prev_loans: errors.append("Active loans cannot exceed total previous loans.")
if amt_annuity >= amt_credit: errors.append("Monthly annuity cannot be larger than the total loan.")

if errors:
    for e in errors: st.error(f"⚠️ {e}")
    st.warning("Please fix input errors to proceed.")
    st.stop()

# ==========================================
# 7. ENGINE: AUTO-CALCULATIONS & PREDICTION
# ==========================================
if st.button("🚀 Run AI Underwriting Engine", type="primary"):
    
    # A. BACKGROUND CALCULATIONS (Your SQL & Phase 3 logic happens invisibly here)
    ext_mean = (ext_1 + ext_2 + ext_3) / 3
    ext_product = ext_1 * ext_2 * ext_3
    
    credit_to_income = amt_credit / amt_income
    annuity_to_income = amt_annuity / amt_income
    active_loan_ratio = active_loans / total_prev_loans if total_prev_loans > 0 else 0
    
    # Skewness Transform
    amt_credit_log = np.log1p(amt_credit)

    # B. BUILD THE EXACT MODEL INPUT DICTIONARY
    input_dict = {
        'AGE': age_years,
        'DAYS_EMPLOYED': days_employed,
        'AMT_INCOME_TOTAL': amt_income,
        'AMT_CREDIT': amt_credit_log,
        'AMT_ANNUITY': amt_annuity,
        'EXT_SOURCE_1': ext_1,
        'EXT_SOURCE_2': ext_2,
        'EXT_SOURCE_3': ext_3,
        'EXT_SOURCE_MEAN': ext_mean,
        'EXT_SOURCE_PRODUCT': ext_product,
        'total_previous_loans': total_prev_loans,
        'active_loans': active_loans,
        'bureau_debt_to_credit_ratio': bureau_debt_ratio,
        'total_installments': total_installments,
        'late_ratio': late_ratio,
        'avg_delay': avg_delay,
        'payment_ratio': payment_ratio,
        'credit_to_income_ratio': credit_to_income,
        'annuity_to_income_ratio': annuity_to_income,
        'active_loan_ratio': active_loan_ratio
    }
    
    # Safely pad missing columns with 0.0 based on what LightGBM expects
    full_input = {feat: input_dict.get(feat, 0.0) for feat in model.feature_name_}
    input_df = pd.DataFrame([full_input])

    # C. PREDICT
    prob = model.predict_proba(input_df)[0][1]
    decision = "REJECTED" if prob > 0.25 else "APPROVED"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("System Decision", f"{'❌' if decision=='REJECTED' else '✅'} {decision}")
    with col2:
        st.metric("Probability of Default", f"{prob:.2%}")

    # D. EXTRACT SHAP & FEED LLM
    with st.spinner("Extracting Mathematical SHAP Evidence..."):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_df)
        shap_vals_default = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        
        # Get Top 5 most impactful features (positive or negative)
        top_indices = np.argsort(np.abs(shap_vals_default))[-5:]
        
        shap_evidence = ""
        for i in top_indices:
            feat_name = input_df.columns[i]
            impact = shap_vals_default[i]
            desc = OFFICIAL_DICTIONARY.get(feat_name, "Internal Risk Metric")
            shap_evidence += f"- {feat_name} | {desc} | Impact: {impact:+.3f}\n"

    with st.spinner("Generating Regulatory-Compliant Letters..."):
        ai_report = generate_ai_feedback(prob, shap_evidence, decision)
        
        st.divider()
        if "### APPLICANT LETTER" in ai_report:
            memo, letter = ai_report.split("### APPLICANT LETTER")
            
            tab1, tab2 = st.tabs(["🔒 Internal Loan Officer Memo", "✉️ Applicant Letter"])
            with tab1:
                st.info(memo.replace("### INTERNAL BANK MEMO", "").strip())
            with tab2:
                st.success(letter.strip())
        else:
            st.write(ai_report)