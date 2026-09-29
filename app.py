"""
Loan Eligibility Pre-Screener (Layer 1: form-based app, no AI yet)
Author: Aman Aggarwal (065064), FORE School of Management
Run locally:  streamlit run app.py
"""
import json
import joblib
import pandas as pd
import streamlit as st

import loan_model as lm   # shared logic: features, validation, policy rules, scoring

st.set_page_config(page_title="Loan Eligibility Pre-Screener", page_icon="🏦", layout="centered")


# ------------------------------------------------------------------
# Load the model once and keep it in memory (not reloaded on every click)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/loan_model.joblib")
    with open("artifacts/config.json") as f:
        cfg = json.load(f)
    return model, cfg

model, cfg = load_model()

# Customer-friendly labels mapped to the codes the model expects
HOME = {"Renting": "RENT", "Own with a mortgage": "MORTGAGE",
        "Own outright": "OWN", "Other": "OTHER"}
PURPOSE = {"Education": "EDUCATION", "Medical expenses": "MEDICAL",
           "Starting or growing a business": "VENTURE", "Personal needs": "PERSONAL",
           "Paying off existing debt": "DEBTCONSOLIDATION", "Home improvement": "HOMEIMPROVEMENT"}
BAND_STYLE = {
    "Likely eligible": ("#1E7B4F", "#E8F5EE",
        "Based on the details you shared, your profile looks like applicants who usually repay on time."),
    "Needs further review": ("#A86400", "#FFF4E0",
        "Your application may still be approved, but a loan officer would need to look at a few details first."),
    "Unlikely at this time": ("#B3261E", "#FDECEA",
        "Based on these details, approval looks unlikely right now. The factors below show what is weighing on the result."),
}

# ------------------------------------------------------------------
# Session state: history of assessments in this browser session
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ------------------------------------------------------------------
# Header and sidebar
# ------------------------------------------------------------------
st.title("Loan Eligibility Pre-Screener")
st.write("Check in under a minute whether you are likely to qualify for a personal loan, "
         "and see which factors matter most. This is an indicative check, **not a loan decision**.")

with st.sidebar:
    st.header("About this tool")
    st.write("A machine-learning model (XGBoost) trained on about 32,000 past loans estimates "
             "the chance of repayment difficulty. Credit-policy rules then make the result more "
             "cautious where needed.")
    m = cfg["test_metrics"]
    st.write(f"**Model accuracy on unseen data:** ROC-AUC {m['ROC-AUC']:.2f}, "
             f"catches {m['Recall']:.0%} of defaulters.")
    st.info("Amounts are in **US dollars** because the model was trained on a US demonstration "
            "dataset. It is not calibrated for Indian borrowers.")
    st.caption("Built by Aman Aggarwal (065064), FORE School of Management.")

# ------------------------------------------------------------------
# Input form (a form submits all fields at once, so there are no partial reruns)
# ------------------------------------------------------------------
with st.form("applicant"):
    st.subheader("Your details")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", key="age", min_value=0, max_value=150, value=30, step=1)
    income = c2.number_input("Annual income (USD)", key="income", min_value=0, max_value=20_000_000,
                             value=50_000, step=1_000)
    emp = c1.number_input("Years in current employment", key="emp", min_value=0.0, max_value=80.0,
                          value=3.0, step=0.5)
    hist = c2.number_input("Length of credit history (years)", key="hist", min_value=0, max_value=80,
                           value=4, step=1)
    home = c1.selectbox("Home ownership", list(HOME), key="home")
    prior = c2.radio("Have you ever defaulted on a loan?", ["No", "Yes"], horizontal=True, key="prior")

    st.subheader("Loan request")
    c3, c4 = st.columns(2)
    amount = c3.number_input("Loan amount (USD)", key="amount", min_value=0, max_value=1_000_000,
                             value=10_000, step=500)
    purpose = c4.selectbox("Purpose of the loan", list(PURPOSE), key="purpose")

    consent = st.checkbox("I understand this is an indicative check and not a credit decision.")
    submitted = st.form_submit_button("Check my eligibility", type="primary",
                                      width="stretch")

# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------
if submitted:
    applicant = dict(
        person_age=int(age), person_income=float(income), person_emp_length=float(emp),
        loan_amnt=float(amount), cb_person_cred_hist_length=float(hist),
        person_home_ownership=HOME[home], loan_intent=PURPOSE[purpose],
        cb_person_default_on_file="Y" if prior == "Yes" else "N",
    )
    if not consent:
        st.warning("Please tick the confirmation box above to continue.")
    elif applicant == st.session_state.last_inputs:
        # Same inputs submitted twice: reuse the result instead of scoring and logging again
        st.caption("These are the same details as your last check, so the result is unchanged.")
    else:
        result = lm.score_applicant(model, applicant, cfg)
        st.session_state.last_inputs = applicant
        st.session_state.last_result = result
        if result["valid"]:
            st.session_state.history.append({
                "Age": int(age), "Income": f"${income:,.0f}", "Loan": f"${amount:,.0f}",
                "Purpose": purpose, "Risk": f"{result['probability_of_default']:.1%}",
                "Result": result["final_band"],
            })

result = st.session_state.last_result
if result is not None:
    st.divider()
    if not result["valid"]:
        st.error("Please correct the following and try again:")
        for e in result["errors"]:
            st.write(f"- {e}")
    else:
        band = result["final_band"]
        color, bg, message = BAND_STYLE[band]
        st.markdown(
            f"""<div style="border-left:6px solid {color};background:{bg};padding:1.1rem 1.3rem;
            border-radius:6px;margin-bottom:1rem">
            <div style="font-size:0.95rem;color:{color};font-weight:600">Your pre-screening result</div>
            <div style="font-size:1.9rem;font-weight:700;color:{color};line-height:1.25">{band}</div>
            <div style="margin-top:0.4rem">{message}</div></div>""",
            unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        p = result["probability_of_default"]
        shown = "Above 90%" if p >= 0.90 else ("Below 2%" if p < 0.02 else f"{p:.1%}")
        col1.metric("Estimated chance of repayment difficulty", shown)
        col2.metric("Average across all past applicants", f"{cfg['base_default_rate']:.1%}")

        if result["policy_reasons"]:
            st.warning("**Credit policy checks applied:**\n\n" +
                       "\n".join(f"- {r}" for r in result["policy_reasons"]))

        st.subheader("What influenced this result")
        drivers = pd.DataFrame(result["top_drivers"], columns=["Factor", "Impact"])
        for _, row in drivers.iterrows():
            direction = "raised your risk" if row.Impact > 0 else "lowered your risk"
            icon = "🔺" if row.Impact > 0 else "🟢"
            st.write(f"{icon} **{row.Factor}** {direction}")
        chart = drivers.set_index("Factor")["Impact"].sort_values()
        st.bar_chart(chart, horizontal=True, color="#0F5257", height=240)
        st.caption("Bars to the right raised the estimated risk; bars to the left lowered it.")

    st.info("This is an automated, indicative pre-screening. Final decisions are made by a "
            "loan officer after full verification. You can always ask to speak to a person.")

# ------------------------------------------------------------------
# History of checks in this session
# ------------------------------------------------------------------
if st.session_state.history:
    with st.expander(f"Your checks this session ({len(st.session_state.history)})"):
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True,
                     width="stretch")
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.last_inputs = None
            st.session_state.last_result = None
            st.rerun()
