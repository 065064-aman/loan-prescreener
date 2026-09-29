"""
loan_model.py
Shared logic for the Loan Eligibility Pre-Screener.
Used by BOTH the training notebook and the Streamlit app, so the app
applies exactly the same feature engineering, validation and policy rules.
Author: Aman Aggarwal (065064), FORE School of Management
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 1. Feature definitions
# ---------------------------------------------------------------
# Inputs a customer can answer at pre-screening time
NUM_INPUTS = ["person_age", "person_income", "person_emp_length",
              "loan_amnt", "cb_person_cred_hist_length"]
CAT_INPUTS = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]

# Assigned by the lender AFTER assessment, so NOT available at pre-screening
LENDER_NUM = ["loan_int_rate"]
LENDER_CAT = ["loan_grade"]

ENGINEERED = ["loan_to_income", "log_income", "log_loan", "emp_to_age",
              "hist_to_age", "high_burden", "prior_default_high_burden"]

CATEGORIES = {
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                    "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
    "cb_person_default_on_file": ["N", "Y"],
}


def add_features(X):
    """Create engineered features. Uses no statistics from the data,
    so it can be applied before or after the train/test split without leakage."""
    X = X.copy()
    X["loan_to_income"] = X["loan_amnt"] / X["person_income"]
    X["log_income"] = np.log1p(X["person_income"])
    X["log_loan"] = np.log1p(X["loan_amnt"])
    X["emp_to_age"] = X["person_emp_length"] / X["person_age"]
    X["hist_to_age"] = X["cb_person_cred_hist_length"] / X["person_age"]
    X["high_burden"] = (X["loan_to_income"] > 0.4).astype(int)
    X["prior_default_high_burden"] = ((X["cb_person_default_on_file"] == "Y")
                                      & (X["loan_to_income"] > 0.3)).astype(int)
    return X


def feature_lists(include_lender_fields=False):
    num = NUM_INPUTS + ENGINEERED + (LENDER_NUM if include_lender_fields else [])
    cat = CAT_INPUTS + (LENDER_CAT if include_lender_fields else [])
    return num, cat


# ---------------------------------------------------------------
# 2. Human-readable names (used for explanations)
# ---------------------------------------------------------------
GROUP_OF = {"log_income": "person_income", "log_loan": "loan_amnt"}
FRIENDLY = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_emp_length": "Years in employment",
    "loan_amnt": "Loan amount requested",
    "cb_person_cred_hist_length": "Length of credit history",
    "loan_to_income": "Loan-to-income ratio",
    "emp_to_age": "Employment stability",
    "hist_to_age": "Credit history relative to age",
    "high_burden": "High repayment burden",
    "prior_default_high_burden": "Past default combined with high burden",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
    "cb_person_default_on_file": "Previous default on record",
    "loan_grade": "Loan grade",
    "loan_int_rate": "Interest rate",
}
ALL_BASE = NUM_INPUTS + CAT_INPUTS + LENDER_NUM + LENDER_CAT + ENGINEERED


def base_feature(transformed_name):
    """Map a transformed column (e.g. 'cat__loan_intent_MEDICAL') to its base input."""
    name = transformed_name.split("__", 1)[-1]
    for col in sorted(ALL_BASE, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return GROUP_OF.get(col, col)
    return name


# ---------------------------------------------------------------
# 3. The deployable model object
# ---------------------------------------------------------------
class LoanRiskModel:
    """Preprocessor + classifier + probability calibrator in one object."""

    def __init__(self, preprocessor, classifier, calibrator, kind, feature_names):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.calibrator = calibrator
        self.kind = kind                    # "xgboost" or "logistic"
        self.feature_names = list(feature_names)

    def _transform(self, X):
        return self.preprocessor.transform(X)

    def raw_score(self, X):
        return self.classifier.predict_proba(self._transform(X))[:, 1]

    def predict_pd(self, X):
        """Calibrated probability of default."""
        return self.calibrator.predict(self.raw_score(X))

    def explain(self, X, top_n=5):
        """Per-applicant contributions (log-odds scale), grouped to base inputs.
        Positive = pushes risk UP, negative = pushes risk DOWN."""
        Xt = self._transform(X)
        if self.kind == "xgboost":
            import xgboost as xgb
            booster = self.classifier.get_booster()
            it = getattr(self.classifier, "best_iteration", None)
            rng = (0, it + 1) if it is not None else (0, 0)
            contrib = booster.predict(xgb.DMatrix(Xt), pred_contribs=True,
                                      iteration_range=rng)[:, :-1]
        elif self.kind == "logistic":
            contrib = Xt * self.classifier.coef_[0]
        else:
            raise NotImplementedError(self.kind)
        out = []
        for row in np.atleast_2d(contrib):
            agg = {}
            for name, v in zip(self.feature_names, row):
                b = base_feature(name)
                agg[b] = agg.get(b, 0.0) + float(v)
            ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
            out.append([(FRIENDLY.get(k, k), round(v, 3)) for k, v in ranked])
        return out


# ---------------------------------------------------------------
# 4. Input validation (hard limits: reject impossible inputs)
# ---------------------------------------------------------------
HARD_LIMITS = {
    "person_age": (18, 100),
    "person_income": (1_000, 10_000_000),
    "person_emp_length": (0, 60),
    "loan_amnt": (500, 35_000),          # product limit in the training data
    "cb_person_cred_hist_length": (0, 60),
}


def validate_applicant(a):
    """Return a list of error messages. Empty list = valid."""
    errors = []
    for f, (lo, hi) in HARD_LIMITS.items():
        v = a.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"{FRIENDLY[f]} is missing.")
        elif not (lo <= v <= hi):
            errors.append(f"{FRIENDLY[f]} must be between {lo:,} and {hi:,} (got {v:,}).")
    for f, allowed in CATEGORIES.items():
        if a.get(f) not in allowed:
            errors.append(f"{FRIENDLY[f]} must be one of {allowed}.")
    if not errors:
        working_years = a["person_age"] - 14
        if a["person_emp_length"] > working_years:
            errors.append("Years in employment cannot exceed age minus 14.")
        if a["cb_person_cred_hist_length"] > working_years:
            errors.append("Credit history cannot be longer than age minus 14.")
    return errors


# ---------------------------------------------------------------
# 5. Bands and policy rules
# ---------------------------------------------------------------
BANDS = ["Likely eligible", "Needs further review", "Unlikely at this time"]


def model_band(pd_value, t_low, t_high):
    if pd_value >= t_high:
        return 2
    if pd_value >= t_low:
        return 1
    return 0


def policy_checks(a, training_ranges=None):
    """Hard credit-policy rules that override the model (they can only make the
    outcome MORE cautious). Returns (minimum_band, list_of_reasons)."""
    reasons, min_band = [], 0
    lti = a["loan_amnt"] / a["person_income"]
    if lti > 0.5:
        min_band = max(min_band, 1)
        reasons.append("Requested loan is more than 50% of annual income.")
    if a["cb_person_default_on_file"] == "Y" and lti > 0.3:
        min_band = max(min_band, 1)
        reasons.append("Previous default on record combined with a loan above 30% of income.")
    if training_ranges:
        for f, (lo, hi) in training_ranges.items():
            if f in a and not (lo <= a[f] <= hi):
                min_band = max(min_band, 1)
                reasons.append(f"{FRIENDLY.get(f, f)} is outside the range the model "
                               f"was trained on, so a human should check this case.")
    return min_band, reasons


def score_applicant(model, a, config):
    """Full decision flow for ONE applicant (dict of raw inputs)."""
    errors = validate_applicant(a)
    if errors:
        return {"valid": False, "errors": errors}
    X = pd.DataFrame([a])
    pd_value = float(model.predict_pd(X)[0])
    mb = model_band(pd_value, config["t_low"], config["t_high"])
    min_band, reasons = policy_checks(a, config.get("training_ranges"))
    final = max(mb, min_band)
    return {
        "valid": True,
        "probability_of_default": round(pd_value, 4),
        "model_band": BANDS[mb],
        "final_band": BANDS[final],
        "policy_override": final != mb,
        "policy_reasons": reasons,
        "top_drivers": model.explain(X)[0],
    }
