import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/gradient_boosting_model.pkl')

st.title("Loan Default Risk Predictor")
st.write("Enter borrower details to estimate default risk.")

# --- Raw inputs, matching the original dataset's columns ---
age = st.number_input("Age", min_value=18, max_value=100, value=35)
monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
debt_ratio = st.number_input("Debt Ratio (debt ÷ income)", min_value=0.0, value=0.3, step=0.01)
utilization = st.number_input("Revolving Credit Utilization (0-1+)", min_value=0.0, value=0.3, step=0.01)
open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", min_value=0, value=5)
real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", min_value=0, value=1)
late_30_59 = st.number_input("Times 30-59 Days Late (last 2 yrs)", min_value=0, value=0)
late_60_89 = st.number_input("Times 60-89 Days Late (last 2 yrs)", min_value=0, value=0)
late_90 = st.number_input("Times 90+ Days Late (last 2 yrs)", min_value=0, value=0)

if st.button("Predict Risk"):
    # --- Replicate feature engineering from Step 5 ---
    total_times_late = late_30_59 + late_60_89 + late_90
    income_per_dependent = monthly_income / (dependents + 1)

    if age < 30:
        age_group = '<30'
    elif age < 45:
        age_group = '30-45'
    elif age < 60:
        age_group = '45-60'
    else:
        age_group = '60+'

    # Build the row exactly matching training column order
    input_dict = {
        'RevolvingUtilizationOfUnsecuredLines': utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
        'NumberOfTimes90DaysLate': late_90,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfDependents': dependents,
        'TotalTimesLate': total_times_late,
        'IncomePerDependent': income_per_dependent,
        'AgeGroup_30-45': 1 if age_group == '30-45' else 0,
        'AgeGroup_45-60': 1 if age_group == '45-60' else 0,
        'AgeGroup_60+': 1 if age_group == '60+' else 0,
    }

    input_df = pd.DataFrame([input_dict])

    proba = model.predict_proba(input_df)[0, 1]
    prediction = "HIGH RISK" if proba >= 0.4 else "LOW RISK"

    st.subheader(f"Result: {prediction}")
    st.write(f"Predicted default probability: **{proba:.1%}**")