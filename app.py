import pandas as pd
import xgboost as xgb
import streamlit as st

model = xgb.Booster()
model.load_model("centseek_model.json")

if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False


def submit_form():
    st.session_state.form_submitted = True


st.header("CentSeek:  Payment History Tracker")

if not st.session_state.form_submitted:
    st.subheader("Payment Delays")
    st.write("Select 0 if paid on time. For delays of 9 or more months, select 9.")
    credit_limit = st.number_input("Credit Limit", min_value=0)
    col1, col2 = st.columns(2)
    with col1:
        pay_0 = st.slider("Month 0", min_value=0, max_value=9, step=1, value=0)
        pay_2 = st.slider("Month 2", min_value=0, max_value=9, step=1, value=0)
        pay_4 = st.slider("Month 4", min_value=0, max_value=9, step=1, value=0)
        pay_6 = st.slider("Month 6", min_value=0, max_value=9, step=1, value=0)
    with col2:
        pay_1 = st.slider("Month 1", min_value=0, max_value=9, step=1, value=0)
        pay_3 = st.slider("Month 3", min_value=0, max_value=9, step=1, value=0)
        pay_5 = st.slider("Month 5", min_value=0, max_value=9, step=1, value=0)

    st.subheader("Bill Amounts")
    bill_cols = st.columns(3)
    with bill_cols[0]:
        bill_amt_3 = st.number_input("Month 3", min_value=0.0, format="%0.2f", key="bill_3")
        bill_amt_6 = st.number_input("Month 6", min_value=0.0, format="%0.2f", key="bill_6")
    with bill_cols[1]:
        bill_amt_1 = st.number_input("Month 1", min_value=0.0, format="%0.2f", key="bill_1")
        bill_amt_4 = st.number_input("Month 4", min_value=0.0, format="%0.2f", key="bill_4")
    with bill_cols[2]:
        bill_amt_2 = st.number_input("Month 2", min_value=0.0, format="%0.2f", key="bill_2")
        bill_amt_5 = st.number_input("Month 5", min_value=0.0, format="%0.2f", key="bill_5")

    st.subheader("Payment Amounts")
    pay_cols = st.columns(3)
    with pay_cols[0]:
        pay_amt_3 = st.number_input("Month 3", min_value=0.0, format="%0.2f", key="pay_3")
        pay_amt_6 = st.number_input("Month 6", min_value=0.0, format="%0.2f", key="pay_6")
    with pay_cols[1]:
        pay_amt_1 = st.number_input("Month 1", min_value=0.0, format="%0.2f", key="pay_1")
        pay_amt_4 = st.number_input("Month 4", min_value=0.0, format="%0.2f", key="pay_4")
    with pay_cols[2]:
        pay_amt_2 = st.number_input("Month 2", min_value=0.0, format="%0.2f", key="pay_2")
        pay_amt_5 = st.number_input("Month 5", min_value=0.0, format="%0.2f", key="pay_5")

    st.button("Submit", on_click=submit_form)
    params = {
        "ID": 0,
        "LIMIT_BAL": credit_limit,
        "PAY_0": pay_amt_1,
        "PAY_2": pay_amt_2,
        "PAY_3": pay_amt_3,
        "PAY_4": pay_amt_4,
        "PAY_5": pay_amt_5,
        "PAY_6": pay_amt_6,
        "BILL_AMT1": bill_amt_1,
        "BILL_AMT2": bill_amt_2,
        "BILL_AMT3": bill_amt_3,
        "BILL_AMT4": bill_amt_4,
        "BILL_AMT5": bill_amt_5,
        "BILL_AMT6": bill_amt_6,
        "PAY_AMT1": pay_amt_1,
        "PAY_AMT2": pay_amt_2,
        "PAY_AMT3": pay_amt_3,
        "PAY_AMT4": pay_amt_4,
        "PAY_AMT5": pay_amt_5,
        "PAY_AMT6": pay_amt_6,
    }
    input_given = pd.DataFrame(data=params, index=[0, 1, 2, 3, 4, 5, 6])
    predictor = xgb.DMatrix(input_given)
    prediction = model.predict(predictor)
    print(prediction[0])
else:
    st.success("Information retrieved successfully")

    st.subheader("Let me take a look - CentSeek")

    if st.button("Enter New Data"):
        st.session_state.form_submitted = False
        st.rerun()
