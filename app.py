import pandas as pd
import xgboost as xgb
import streamlit as st
import traceback
import plotly.graph_objects as go
# Wrap sentiment import in try-except to handle potential errors
try:
    from sentiment import sentiment_analysis, public_sentiment
except ImportError as e:
    print(f"Error importing sentiment module: {e}")


    def sentiment_analysis():
        return {"label": "NEUTRAL", "score": 0.5, "message": "Sentiment analysis module not available"}


    def public_sentiment():
        return "Sentiment analysis module not available"

model = xgb.Booster()
try:
    model.load_model("centseek_model.json")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    st.error(f"Error loading model: {e}")
    model = None

if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# Initialize prediction in session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Initialize params in session state to store form data
if 'params' not in st.session_state:
    st.session_state.params = None


def eval_risk(p_default):
    """Evaluate risk based on default probability"""
    msg = ""
    if 0 <= p_default < 0.4:
        msg = "Very unlikely to default"
    elif 0.4 <= p_default < 0.5:
        msg = "Moderate risk of default"
    else:
        msg = "Very likely to default"

    st.metric("Default Probability", f"{p_default:.2%}", msg)


def plot_feature_importance():
    if model is None:
        st.error("Model not available for feature importance analysis")
        return

    importance = model.get_score(importance_type='gain')  # Get feature importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    fig = go.Figure(go.Bar(
        x=[x[1] for x in sorted_importance],
        y=[x[0] for x in sorted_importance],
        orientation='h',
        marker=dict(color="blue")
    ))
    fig.update_layout(title="Top 10 Feature Importances", xaxis_title="Importance Score")
    st.plotly_chart(fig)


def plot_radar_chart(customer_data):
    if not customer_data:
        st.error("No customer data available for radar chart")
        return

    # Define categories for radar chart
    categories = ["Credit Limit", "Avg Bill Amount", "Avg Payment Amount", "Max Payment Delay"]

    # Calculate the values with proper error handling
    try:
        # Calculate average bill amount
        bill_amounts = [
            float(customer_data.get(f"BILL_AMT{i}", 0))
            for i in range(1, 7)
        ]
        avg_bill = sum(bill_amounts) / len(bill_amounts) if len(bill_amounts) > 0 else 0

        # Calculate average payment amount
        payment_amounts = [
            float(customer_data.get(f"PAY_AMT{i}", 0))
            for i in range(1, 7)
        ]
        avg_payment = sum(payment_amounts) / len(payment_amounts) if len(payment_amounts) > 0 else 0

        # Find maximum payment delay
        payment_delays = [
            int(customer_data.get(f"PAY_{i}", 0))
            for i in range(7)
            if f"PAY_{i}" in customer_data
        ]
        max_delay = max(payment_delays) if payment_delays else 0

        # Get credit limit
        credit_limit = float(customer_data.get("LIMIT_BAL", 0))

        # Create normalized values for better visualization
        # Scale values to a range that works well for radar chart
        max_credit_limit = 100000  # Assumed maximum credit limit for scaling

        values = [
            min(credit_limit / max_credit_limit * 10, 10),  # Scale credit limit
            min(avg_bill / max_credit_limit * 10, 10),  # Scale average bill
            min(avg_payment / max_credit_limit * 10, 10),  # Scale average payment
            min(max_delay, 9)  # Payment delay already in range 0-9
        ]

        # Create the radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name="Customer Risk Profile",
            marker=dict(color="rgba(31, 119, 180, 0.8)"),
            line=dict(color="rgba(31, 119, 180, 1)")
        ))

        # Update layout with better formatting
        fig.update_layout(
            title="Customer Risk Profile",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True
        )

        # Display summary statistics alongside the chart
        col1, col2 = st.columns([3, 2])

        with col1:
            st.plotly_chart(fig)

        with col2:
            st.subheader("Risk Profile Summary")
            st.write(f"Credit Limit: ${credit_limit:,.2f}")
            st.write(f"Average Monthly Bill: ${avg_bill:,.2f}")
            st.write(f"Average Monthly Payment: ${avg_payment:,.2f}")
            st.write(f"Maximum Payment Delay: {max_delay} month(s)")

            # Calculate and display bill-to-payment ratio
            if avg_payment > 0:
                bill_payment_ratio = avg_bill / avg_payment
                st.write(f"Bill-to-Payment Ratio: {bill_payment_ratio:.2f}")

                if bill_payment_ratio > 2:
                    st.warning("⚠️ Bill amounts significantly exceed payments")
                elif bill_payment_ratio < 0.8:
                    st.success("✅ Payments exceed bill amounts")

    except Exception as e:
        st.error(f"Error generating radar chart: {e}")
        st.code(traceback.format_exc())


def run_sentiment_analysis():
    """Run and display sentiment analysis results"""
    try:
        with st.spinner("Analyzing market sentiment..."):
            result = sentiment_analysis()

        # Display sentiment message
        st.info(f"**Market Sentiment**: {result.get('message', 'No message available')}")

        # Display sentiment score with appropriate color
        sentiment_score = result.get('score', 0.5)
        sentiment_label = result.get('label', 'NEUTRAL')

        # Create color-coded display
        color = "green" if sentiment_label == "POSITIVE" else "red" if sentiment_label == "NEGATIVE" else "gray"
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: {color}20;">
            <h3 style="color: {color};">{sentiment_label}</h3>
            <p>Confidence: {sentiment_score:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        st.code(traceback.format_exc())


def submit_form():
    """Set form submitted state to True"""
    st.session_state.form_submitted = True


st.sidebar.markdown("""
    ### Project Description

    This application uses XGBoost to classify data from the UCI Machine Learning Repository. 
    The model is optimized with hyperparameters specifically tuned for binary classification tasks.

    Key features:
    - GPU-accelerated training
    - Optimized hyperparameters
    - Interactive visualization
    - Real-time prediction capabilities
    """)

# Horizontal line for visual separation
st.sidebar.markdown("---")

# GitHub repository link
st.sidebar.markdown("### Project Repository")
git_hub_repo = "https://github.com/YazRaso/Centsi.git"
st.sidebar.markdown(f"[View source code]({git_hub_repo})")

# Footer with love message
st.sidebar.markdown("---")
st.sidebar.markdown("Made with 🩸, 💦, 💧in Waterloo")

st.header("CentSeek: Payment History Tracker")

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
    # Change these lines in your bill amount inputs
    with bill_cols[0]:
        bill_amt_1 = st.number_input("Month 1", min_value=0.0, format="%0.2f", key="BILL_AMT1")
        bill_amt_4 = st.number_input("Month 4", min_value=0.0, format="%0.2f", key="BILL_AMT4")
    with bill_cols[1]:
        bill_amt_2 = st.number_input("Month 2", min_value=0.0, format="%0.2f", key="BILL_AMT2")
        bill_amt_5 = st.number_input("Month 5", min_value=0.0, format="%0.2f", key="BILL_AMT5")
    with bill_cols[2]:
        bill_amt_3 = st.number_input("Month 3", min_value=0.0, format="%0.2f", key="BILL_AMT3")
        bill_amt_6 = st.number_input("Month 6", min_value=0.0, format="%0.2f", key="BILL_AMT6")

    # And similarly for payment amounts
    st.subheader("Amounts Paid")
    pay_cols = st.columns(3)
    with pay_cols[0]:
        pay_amt_1 = st.number_input("Month 1", min_value=0.0, format="%0.2f", key="PAY_AMT1")
        pay_amt_4 = st.number_input("Month 4", min_value=0.0, format="%0.2f", key="PAY_AMT4")
    with pay_cols[1]:
        pay_amt_2 = st.number_input("Month 2", min_value=0.0, format="%0.2f", key="PAY_AMT2")
        pay_amt_5 = st.number_input("Month 5", min_value=0.0, format="%0.2f", key="PAY_AMT5")
    with pay_cols[2]:
        pay_amt_3 = st.number_input("Month 3", min_value=0.0, format="%0.2f", key="PAY_AMT3")
        pay_amt_6 = st.number_input("Month 6", min_value=0.0, format="%0.2f", key="PAY_AMT6")
    if st.button("Submit"):
        params = {
            "LIMIT_BAL": float(credit_limit),
            "PAY_0": float(pay_0),
            # Intentionally omitting PAY_1 as requested
            "PAY_2": float(pay_2),
            "PAY_3": float(pay_3),
            "PAY_4": float(pay_4),
            "PAY_5": float(pay_5),
            "PAY_6": float(pay_6),
            "BILL_AMT1": float(bill_amt_1),
            "BILL_AMT2": float(bill_amt_2),
            "BILL_AMT3": float(bill_amt_3),
            "BILL_AMT4": float(bill_amt_4),
            "BILL_AMT5": float(bill_amt_5),
            "BILL_AMT6": float(bill_amt_6),
            "PAY_AMT1": float(pay_amt_1),
            "PAY_AMT2": float(pay_amt_2),
            "PAY_AMT3": float(pay_amt_3),
            "PAY_AMT4": float(pay_amt_4),
            "PAY_AMT5": float(pay_amt_5),
            "PAY_AMT6": float(pay_amt_6),
        }

        # Store params in session state for radar chart
        st.session_state.params = params

        # First check if model is loaded
        if model is None:
            st.error("Model not loaded. Cannot make predictions.")
            st.session_state.prediction = None
        else:
            try:
                # Make prediction
                input_df = pd.DataFrame(params, index=[0])
                predictor = xgb.DMatrix(input_df)
                prediction_result = model.predict(predictor)
                if len(prediction_result) > 0:
                    bill_sum = 0
                    paid_sum = 0
                    for key, amt in st.session_state.params.items():
                        if key[:4] == "BILL":
                            bill_sum += amt
                        elif len(key) >= 5 and key[:7] == "PAY_AMT":
                            paid_sum += amt
                    if bill_sum <= paid_sum:
                        st.session_state.prediction = 0
                    else:
                        st.session_state.prediction = float(prediction_result[0])
                else:
                    st.error("Prediction returned empty result")
                    st.session_state.prediction = None

                # Submit form after prediction is made
                submit_form()
                st.rerun()
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.code(traceback.format_exc())
                st.session_state.prediction = None
else:
    st.success("Information retrieved successfully")
    st.subheader("Let's see here")

    # Fixed: Check if prediction exists and is not None
    if st.session_state.prediction is not None:
        eval_risk(st.session_state.prediction)

        # Fixed: proper metric options and function mapping
        metric = st.selectbox(
            "What metrics would you like to view?",
            ("Risk Profile", "Feature Importance", "Sentiment Analysis"),
            placeholder="Awaiting orders!",
        )

        # Call the appropriate function based on selection
        if metric == "Sentiment Analysis":
            run_sentiment_analysis()
        elif metric == "Feature Importance":
            plot_feature_importance()
        elif metric == "Risk Profile":
            if st.session_state.params:
                plot_radar_chart(st.session_state.params)
            else:
                st.error("No customer data available for risk profile")
    else:
        st.warning("No prediction data available. Please submit the form again.")

    if st.button("Enter New Data"):
        st.session_state.form_submitted = False
        st.rerun()
