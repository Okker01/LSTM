import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==========================================================
# SETTINGS
# ==========================================================
CSV_FILE = "/mnt/data/BBL_BK_OHLCV_2019-2025.csv"
MODEL_NAME = "lstm_bbl_5days.h5"
SCALER_NAME = "bbl_scaler.save"
SEQ_LEN = 5

# ==========================================================
# LOAD SCALER (joblib)
# ==========================================================
def load_scaler():
    scaler = joblib.load(SCALER_NAME)
    print("Scaler loaded from:", SCALER_NAME)
    return scaler

# ==========================================================
# LOAD TEST DATA (2023–2024)
# ==========================================================
def load_test_data():
    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # test period
    df = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2024-12-31")]

    df_raw = df.copy()
    df = df[["Open", "High", "Low", "Close", "Volume"]].fillna(method="ffill")

    return df_raw, df

# ==========================================================
# PREDICT TOMORROW
# ==========================================================
def predict_tomorrow(model, scaler, df):
    last_5 = df.tail(SEQ_LEN).values
    scaled_input = scaler.transform(last_5).reshape(1, SEQ_LEN, 5)

    pred_scaled = model.predict(scaled_input, verbose=0)[0][0]

    pred_close = scaler.inverse_transform(
        np.array([[0, 0, 0, pred_scaled, 0]])
    )[0][3]

    return pred_close

# ==========================================================
# PLOT FULL 2023–2024 PREDICTIONS
# ==========================================================
def generate_plot_data(model, scaler, df, df_raw):
    scaled = scaler.transform(df)

    X = []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])

    X = np.array(X)
    pred_scaled = model.predict(X, verbose=0)

    pred = []
    for p in pred_scaled:
        inv = scaler.inverse_transform([[0,0,0,p[0],0]])[0][3]
        pred.append(inv)

    actual = df_raw["Close"].iloc[SEQ_LEN:].values
    dates = df_raw["Date"].iloc[SEQ_LEN:].values

    return dates, actual, pred

# ==========================================================
# PLOT
# ==========================================================
def plot_predictions(dates, actual, preds):
    plt.figure(figsize=(14,6))
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

    df_raw, df = load_test_data()

    # 1) Next-day prediction after 2024
    tomorrow = predict_tomorrow(model, scaler, df)
    print("\n=================================")
    print("📌 NEXT-DAY PREDICTED CLOSE =", tomorrow)
    print("=================================\n")

    # 2) Full prediction curve
    dates, actual, preds = generate_plot_data(model, scaler, df, df_raw)
    plot_predictions(dates, actual, preds)

if __name__ == "__main__":
    main()
