"""model.py - feature engineering, training and evaluation (regression only)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 100  # need enough days to build 20-day features and still train

FEATURE_INFO = {
    "Return_1d":     "Today's % change",
    "Return_2d":     "% change over last 2 days",
    "Return_5d":     "% change over last 5 days (short-term momentum)",
    "Close_vs_MA5":  "Close vs 5-day average (short-term trend)",
    "Close_vs_MA20": "Close vs 20-day average (longer trend)",
    "Volatility_10d": "How jumpy the last 10 days were",
    "Range_pct":     "Today's High-Low range as % of Close",
    "Volume_vs_avg": "Today's volume vs 20-day average",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw Open/High/Low/Close/Volume into simple, meaningful features."""
    close = df["Close"]
    ret = close.pct_change()
    f = pd.DataFrame(index=df.index)
    f["Return_1d"] = ret
    f["Return_2d"] = close.pct_change(2)
    f["Return_5d"] = close.pct_change(5)
    f["Close_vs_MA5"] = close / close.rolling(5).mean() - 1
    f["Close_vs_MA20"] = close / close.rolling(20).mean() - 1
    f["Volatility_10d"] = ret.rolling(10).std()
    f["Range_pct"] = (df["High"] - df["Low"]) / close
    f["Volume_vs_avg"] = df["Volume"] / df["Volume"].rolling(20).mean() - 1
    return f


def train_and_evaluate(df: pd.DataFrame, model_name="Linear Regression", test_size=0.2):
    if len(df) < MIN_ROWS:
        raise ValueError(f"Not enough data: got {len(df)} days, need at least {MIN_ROWS}. "
                         "Choose a longer date range.")

    feats = add_features(df)
    # Target: tomorrow's % change in closing price
    target = df["Close"].shift(-1) / df["Close"] - 1

    latest_features = feats.iloc[[-1]]                 # today -> used for tomorrow's forecast
    data = feats.assign(Target=target).dropna()        # drop warm-up rows + last row
    X, y = data[list(FEATURE_INFO)], data["Target"]

    # Chronological split (no shuffling)
    split = int(len(X) * (1 - test_size))
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    reg = LinearRegression() if model_name == "Linear Regression" else Ridge(alpha=10.0)
    model = make_pipeline(StandardScaler(), reg)
    model.fit(X_train, y_train)

    # Predicted % change -> predicted price
    close_today = df["Close"].loc[X_test.index]
    pred_ret = model.predict(X_test)
    pred_price = close_today * (1 + pred_ret)
    actual_price = close_today * (1 + y_test)
    baseline_price = close_today                      # naive: "tomorrow = today"

    def scores(pred):
        return {"MAE": mean_absolute_error(actual_price, pred),
                "RMSE": float(np.sqrt(mean_squared_error(actual_price, pred))),
                "R2": r2_score(actual_price, pred)}

    model_scores, base_scores = scores(pred_price), scores(baseline_price)
    direction = float((np.sign(pred_ret) == np.sign(y_test)).mean() * 100)

    next_ret = float(model.predict(latest_features)[0])
    last_close = float(df["Close"].iloc[-1])

    weights = pd.Series(reg.coef_, index=list(FEATURE_INFO)).sort_values()
    results = pd.DataFrame({"Actual": actual_price, "Predicted": pred_price,
                            "Baseline": baseline_price}, index=X_test.index)
    return {
        "model_scores": model_scores, "base_scores": base_scores,
        "direction_acc": direction, "results": results, "weights": weights,
        "next_price": last_close * (1 + next_ret), "next_return": next_ret * 100,
        "last_close": last_close, "last_date": df.index[-1],
        "n_train": len(X_train), "n_test": len(X_test),
    }