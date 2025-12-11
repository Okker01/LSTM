import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load scaler info
scaler_data = np.load("bbl_scaler.npy", allow_pickle=True).item()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.min_ = scaler_data["min_"]
scaler.scale_ = scaler_data["scale_"]
scaler.data_min_ = scaler_data["data_min_"]
scaler.data_max_ = scaler_data["data_max_"]
scaler.data_range_ = scaler_data["data_range_"]


# ==========================================================
# SETTINGS
# ==========================================================
CSV_FILE = "BBL_BK_OHLCV_2019-2025.csv"
MODEL_NAME = "lstm_bbl_5days.h5"
SEQ_LEN = 5


# ==========================================================
# LOAD MODEL
# ==========================================================
print("Loading LSTM model:", MODEL_NAME)
model = load_model(MODEL_NAME)


# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(CSV_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Use OHLCV features only
df = df[["Open", "High", "Low", "Close", "Volume"]]

df = df.fillna(method="ffill")


# ==========================================================
# TAKE LAST 5 DAYS AS INPUT
# ==========================================================
last_5_days = df.tail(SEQ_LEN).values
scaled_input = scaler.transform(last_5_days)

X_input = np.array([scaled_input])  # shape (1,5,5)


# ==========================================================
# MAKE PREDICTION
# ==========================================================
scaled_pred = model.predict(X_input)
pred_close = scaler.inverse_transform(
    np.array([np.array([0,0,0,scaled_pred[0][0],0])])
)[0][3]

print("\n==============================")
print("📌 PREDICTED NEXT-DAY CLOSE =", pred_close)
print("==============================")

