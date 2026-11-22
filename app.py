"""app.py - simple Streamlit UI. Run with: streamlit run app.py"""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_from_csv, load_from_yahoo
from model import FEATURE_INFO, train_and_evaluate

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="centered")
st.title("📈 Next-Day Stock Price Predictor")
st.caption("A regression model that estimates tomorrow's closing price. "
           "It is a learning project, not financial advice.")

# ------------------------- Step 1: choose data -------------------------
st.header("1. Choose your data")
source = st.radio("Data source", ["Yahoo Finance", "Upload CSV files (NIFTY 50)"], horizontal=True)

df, label = None, ""
try:
    if source == "Yahoo Finance":
        c1, c2, c3 = st.columns([2, 2, 2])
        ticker = c1.text_input("Ticker", "AAPL", help="e.g. AAPL, MSFT, RELIANCE.NS")
        start = c2.date_input("From", date.today() - timedelta(days=3 * 365))
        end = c3.date_input("To", date.today())
        if start >= end:
            st.error("'From' date must be before 'To' date.")
        else:
            with st.spinner("Downloading data..."):
                df = load_from_yahoo(ticker, start, end)
            label = ticker.upper()
    else:
        files = st.file_uploader("Upload one or more NIFTY CSV files", type="csv",
                                 accept_multiple_files=True)
        if files:
            df = pd.concat([load_from_csv(f) for f in files]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            label = "NIFTY 50"
except ValueError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Something went wrong while loading data: {e}")

if df is None or df.empty:
    st.info("Pick a stock or upload CSV files to continue.")
    st.stop()

st.success(f"Loaded **{label}**: {len(df)} trading days "
           f"({df.index.min().date()} to {df.index.max().date()})")
fig = go.Figure(go.Scatter(x=df.index, y=df["Close"], line=dict(width=2)))
fig.update_layout(title="Closing price history", height=300, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig)
with st.expander("View data table"):
    st.dataframe(df.sort_index(ascending=False))

# ------------------------- Step 2: train -------------------------
st.header("2. Train the model")
model_name = st.selectbox("Model", ["Linear Regression", "Ridge Regression"])
if st.button("Train model", type="primary"):
    try:
        st.session_state["out"] = train_and_evaluate(df, model_name)
        st.session_state["label"] = label
    except ValueError as e:
        st.error(str(e))

out = st.session_state.get("out")
if not out or st.session_state.get("label") != label:
    st.stop()

# ------------------------- Step 3: results -------------------------
st.header("3. Prediction")
change = out["next_price"] - out["last_close"]
a, b = st.columns(2)
a.metric(f"Last close ({out['last_date'].date()})", f"{out['last_close']:,.2f}")
b.metric("Predicted next close", f"{out['next_price']:,.2f}",
         delta=f"{change:+,.2f} ({out['next_return']:+.2f}%)")
st.caption("A statistical estimate only. Real prices depend on news and events the model cannot see.")

st.header("4. How good is it?")
st.write(f"Tested on the most recent **{out['n_test']} days** the model never saw during training "
         f"(trained on the earlier {out['n_train']}).")
m, base = out["model_scores"], out["base_scores"]
table = pd.DataFrame({
    "Our model": [m["MAE"], m["RMSE"], m["R2"]],
    "Naive guess (tomorrow = today)": [base["MAE"], base["RMSE"], base["R2"]],
}, index=["MAE (avg error, lower is better)", "RMSE (lower is better)", "R² (higher is better)"])
st.dataframe(table.style.format("{:,.3f}"))

d1, d2 = st.columns(2)
d1.metric("Direction correct", f"{out['direction_acc']:.1f}%",
          help="How often the model got the up/down move right. Coin flip = 50%.")
better = m["MAE"] < base["MAE"]
d2.metric("Beats naive guess (MAE)?", "Yes" if better else "No")
if abs(out["direction_acc"] - 50) < 8:
    st.info("Direction accuracy is close to a coin flip on this test period, "
            "which is typical: daily stock moves are mostly noise.")

st.subheader("Actual vs predicted (test period)")
res = out["results"]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=res.index, y=res["Actual"], name="Actual", line=dict(width=2)))
fig2.add_trace(go.Scatter(x=res.index, y=res["Predicted"], name="Predicted",
                          line=dict(width=2, dash="dot")))
fig2.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                   legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig2)

# ------------------------- Explanation -------------------------
with st.expander("What does the model look at?"):
    st.write("The model predicts tomorrow's **% change**, then converts it to a price. "
             "Bars show how each feature pushes the prediction up (right) or down (left).")
    w = out["weights"]
    fig3 = go.Figure(go.Bar(x=w.values, y=w.index, orientation="h"))
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3)
    for k, v in FEATURE_INFO.items():
        st.markdown(f"- **{k}**: {v}")