import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ==========================================================
# SETTINGS
# ==========================================================
CSV_FILE = "/mnt/data/BBL_BK_OHLCV_2019-2025.csv"
MODEL_NAME = "lstm_bbl_5days.h5"
SCALER_NAME = "bbl_scaler.save"
SEQ_LEN = 5

# ==========================================================
# LOAD SCALER
# ==========================================================
def load_scaler():
    if not os.path.exists(SCALER_NAME):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_NAME}")
    scaler = joblib.load(SCALER_NAME)
    print("Scaler loaded:", SCALER_NAME)
    return scaler

# ==========================================================
# LOAD TEST DATA (2023–2024)
# ==========================================================
def load_test_data():
    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Test period 2023 → 2024
    df_test = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2024-12-31")]

    df_raw = df_test.copy()
    df_clean = df_test[["Open", "High", "Low", "Close", "Volume"]].fillna(method="ffill")

    return df_raw, df_clean

# ==========================================================
# PREDICT TOMORROW
# ==========================================================
def predict_tomorrow(model, scaler, df_clean):
    last_5 = df_clean.tail(SEQ_LEN).values
    scaled_input = scaler.transform(last_5).reshape(1, SEQ_LEN, 5)

    pred_scaled = model.predict(scaled_input, verbose=0)[0][0]

    pred_close = scaler.inverse_transform(
        np.array([[0, 0, 0, pred_scaled, 0]])
    )[0][3]

    return pred_close

# ==========================================================
# GENERATE FULL TEST PREDICTIONS
# ==========================================================
def generate_plot_data(model, scaler, df_clean, df_raw):
    scaled = scaler.transform(df_clean)

    X = []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])

    X = np.array(X)
    pred_scaled = model.predict(X, verbose=0)

    preds = []
    for p in pred_scaled:
        inv = scaler.inverse_transform([[0, 0, 0, p[0], 0]])[0][3]
        preds.append(inv)

    actual = df_raw["Close"].iloc[SEQ_LEN:].values
    dates = df_raw["Date"].iloc[SEQ_LEN:].values

    return dates, actual, preds

# ==========================================================
# PLOT PREDICTIONS
# ==========================================================
def plot_predictions(dates, actual, preds):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label="Actual Close")
    plt.plot(dates, preds, label="Predicted Close", alpha=0.8)
    plt.title("BBL Close Price Prediction – Test Period 2023–2024")
    plt.xlabel("Date")
    plt.ylabel("Close Price (THB)")
    plt.grid(True)
    plt.legend()
    plt.show()

# ==========================================================
# MAIN
# ==========================================================
def main():
    scaler = load_scaler()
    model = load_model(MODEL_NAME)

    df_raw, df_clean = load_test_data()

    # 🔥 Predict tomorrow after last 2024 data
    tomorrow = predict_tomorrow(model, scaler, df_clean)
    print("\n=================================")
    print("📌 NEXT-DAY PREDICTED CLOSE =", tomorrow)
    print("=================================\n")

    # 🔥 Full prediction chart
    dates, actual, preds = generate_plot_data(model, scaler, df_clean, df_raw)
    plot_predictions(dates, actual, preds)

if __name__ == "__main__":
    main()
