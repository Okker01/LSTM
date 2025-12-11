import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


# ==========================================================
# GPU CHECK
# ==========================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("🟢 GPU detected and enabled")
else:
    print("⚠️ No GPU found — running on CPU")


# ==========================================================
# SETTINGS
# ==========================================================
CSV_FILE = "BBL_BK_OHLCV_2019-2025.csv"
SEQ_LEN = 5                     # USE PAST 5 DAYS
MODEL_NAME = "lstm_bbl_5days.h5"
SCALER_NAME = "bbl_scaler.npy"


# ==========================================================
# 1. LOAD DATA
# ==========================================================
def load_data():
    print("Loading dataset:", CSV_FILE)
    df = pd.read_csv(CSV_FILE)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Select OHLCV features
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df = df.fillna(method="ffill")
    return df


# ==========================================================
# 2. PREPROCESS
# ==========================================================
def preprocess(df):
    print("Scaling OHLCV values...")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Save scaler parameters
    np.save(SCALER_NAME, {
        "min_": scaler.min_,
        "scale_": scaler.scale_,
        "data_min_": scaler.data_min_,
        "data_max_": scaler.data_max_,
        "data_range_": scaler.data_range_
    })

    X, y = [], []

    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])  # 5-day sequence
        y.append(scaled[i][3])         # next-day close (index 3)

    return np.array(X), np.array(y), scaler


# ==========================================================
# 3. BUILD MODEL
# ==========================================================
def build_model(input_shape):

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),

        LSTM(32),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1)  # predict next day close
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
def train_model(model, X, y):
    es = EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        shuffle=True,
        callbacks=[es]
    )

    return history


# ==========================================================
# 5. PLOT LOSS
# ==========================================================
def plot_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


# ==========================================================
# MAIN
# ==========================================================
def main():
    df = load_data()
    X, y, scaler = preprocess(df)

    model = build_model((X.shape[1], X.shape[2]))

    history = train_model(model, X, y)
    plot_loss(history)

    model.save(MODEL_NAME)
    print("\nModel saved:", MODEL_NAME)
    print("Scaler saved:", SCALER_NAME)


# ==========================================================
if __name__ == "__main__":
    main()
