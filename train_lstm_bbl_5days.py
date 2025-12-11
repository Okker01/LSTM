import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
import os

# ==========================================================
# GPU CHECK
# ==========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    print("🟢 GPU enabled")
else:
    print("⚠️ No GPU found")

# ==========================================================
# SETTINGS
# ==========================================================
CSV_FILE = "/mnt/data/BBL_BK_OHLCV_2019-2025.csv"
SEQ_LEN = 5

SAVE_DIR = "bbl_lstm"          # 🔥 save folder
MODEL_NAME = f"{SAVE_DIR}/lstm_bbl_5days.h5"
SCALER_NAME = f"{SAVE_DIR}/bbl_scaler.save"

# create folder if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================================
# 1. LOAD & CLEAN DATA (TRAIN 2019–2023)
# ==========================================================
def load_data():
    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    df_train = df[(df["Date"].dt.year >= 2019) & (df["Date"].dt.year <= 2023)]

    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_train[num_cols] = df_train[num_cols].apply(pd.to_numeric, errors="coerce")
    df_train[num_cols] = df_train[num_cols].fillna(method="ffill")

    return df_train[["Date"] + num_cols]

# ==========================================================
# 2. PREPROCESS
# ==========================================================
def preprocess(df):
    features = ["Open", "High", "Low", "Close", "Volume"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    joblib.dump(scaler, SCALER_NAME)
    print("Scaler saved:", SCALER_NAME)

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i][3])

    return np.array(X), np.array(y), scaler

# ==========================================================
# 3. MODEL
# ==========================================================
def build_model(shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model

# ==========================================================
# MAIN
# ==========================================================
def main():
    df = load_data()
    X, y, _ = preprocess(df)

    model = build_model((X.shape[1], X.shape[2]))

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        callbacks=[es]
    )

    # Save model into folder
    model.save(MODEL_NAME)
    print("Model saved to:", MODEL_NAME)

if __name__ == "__main__":
    main()
