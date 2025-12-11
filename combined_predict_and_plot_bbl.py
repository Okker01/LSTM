import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# LOAD SCALER
# ==========================================================
def load_scaler():
    scaler_data = np.load("bbl_scaler.npy", allow_pickle=True).item()

    scaler = MinMaxScaler()
    scaler.min_ = scaler_data["min_"]
    scaler.scale_ = scaler_data["scale_"]
    scaler.data_min_ = scaler_data["data_min_"]
    scaler.data_max_ = scaler_data["data_max_"]
    scaler.data_range_ = scaler_data["data_range_"]

    return scaler


# ==========================================================
# SETTINGS
# ==========================================================
CSV_FILE = "BBL_BK_OHLCV_2019-2025.csv"
MODEL_NAME = "lstm_bbl_5days.h5"
SEQ_LEN = 5


# ==========================================================
# LOAD DATA
# ==========================================================
def load_data():
    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df_raw = df.copy()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.fillna(method="ffill")

    return df_raw, df


# ==========================================================
# PREDICT TOMORROW
# ==========================================================
def predict_tomorrow(model, scaler, df):
    last_5 = df.tail(SEQ_LEN).values
    scaled_input = scaler.transform(last_5)

    X_input = np.array([scaled_input])  # shape (1,5,5)

    scaled_pred = model.predict(X_input)

    # inverse transform only CLOSE price
    pred_close = scaler.inverse_transform(
        np.array([[0, 0, 0, scaled_pred[0][0], 0]])
    )[0][3]

    return pred_close


# ==========================================================
# MAKE FULL PREDICTIONS FOR PLOT
# ==========================================================
def generate_plot_predictions(model, scaler, df, df_raw):
    scaled = scaler.transform(df)

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i][3])

    X = np.array(X)
    y = np.array(y)

    scaled_preds = model.predict(X)

    inv_preds = []
    for p in scaled_preds:
        inv = scaler.inverse_transform(
            np.array([[0, 0, 0, p[0], 0]])
        )[0][3]
        inv_preds.append(inv)

    actual = df_raw["Close"].iloc[SEQ_LEN:].values
    dates = df_raw["Date"].iloc[SEQ_LEN:].values

    return dates, actual, inv_preds


# ==========================================================
# PLOT FUNCTION
# ==========================================================
def plot_predictions(dates, actual, preds):
    plt.figure(figsize=(14,6))
    plt.plot(dates, actual, label="Actual Close")
    plt.plot(dates, preds, label="Predicted Close", alpha=0.8)
    plt.title("Actual vs Predicted Close Price (BBL)")
    plt.xlabel("Date")
    plt.ylabel("Price (THB)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================================
# MAIN SCRIPT
# ==========================================================
def main():
    # Load everything
    scaler = load_scaler()
    model = load_model(MODEL_NAME)
    df_raw, df = load_data()

    # 1) Predict Tomorrow
    tomorrow_pred = predict_tomorrow(model, scaler, df)

    print("\n=====================================")
    print("📌 Predicted NEXT-DAY Close Price =", tomorrow_pred)
    print("=====================================\n")

    # 2) Generate full prediction plot
    dates, actual, preds = generate_plot_predictions(model, scaler, df, df_raw)
    plot_predictions(dates, actual, preds)


# Run
if __name__ == "__main__":
    main()
