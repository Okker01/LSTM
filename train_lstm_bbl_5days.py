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
# FOLDER TO SAVE MODEL + SCALER
# ==========================================================
SAVE_FOLDER = "saved_models"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==========================================================
# GPU CHECK
# ==========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print("🟢 GPU detected and enabled")
else:
    print("⚠️ No GPU found — running on CPU")

# ==========================================================
# SETTINGS
# ==========================================================
CSV_FILE = "/mnt/data/BBL_BK_OHLCV_2019-2025.csv"
SEQ_LEN = 5
MODEL_NAME = "lstm_bbl_5days.h5"
SCALER_NAME = "bbl_scaler.save"

# ==========================================================
# 1. LOAD & CLEAN DATA
# ==========================================================
def load_data():
    print("Loading dataset:", CSV_FILE)
    df = pd.read_csv(CSV_FILE)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df = df.sort_values("Date")

    df_train = df[(df["Date"].dt.year >= 2019) & (df["Date"].dt.year <= 2023)]
    print("Training period:", df_train["Date"].min().date(), "to", df_train["Date"].max().date())
    print("Training rows:", len(df_train))

    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_train[num_cols] = df_train[num_cols].apply(pd.to_numeric, errors="coerce")
    df_train[num_cols] = df_train[num_cols].fillna(method="ffill").fillna(method="bfill")

    return df_train[["Date"] + num_cols]

# ==========================================================
# 2. PREPROCESS FOR LSTM
# ==========================================================
def preprocess(df):
    features = ["Open", "High", "Low", "Close", "Volume"]

    print("Scaling OHLCV values...")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features].values)

    # Save scaler in folder
    joblib.dump(scaler, os.path.join(SAVE_FOLDER, SCALER_NAME))
    print("Scaler saved to:", os.path.join(SAVE_FOLDER, SCALER_NAME))

    X, y = [], []

    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i][3])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1), scaler

# ==========================================================
# 3. BUILD MODEL
# ==========================================================
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model

# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
def train_model(model, X, y):
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    return model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        shuffle=False,
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )

# ==========================================================
# 5. PLOT LOSS
# ==========================================================
def plot_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# ==========================================================
# 6. MAIN
# ==========================================================
def main():
    df = load_data()
    X, y, scaler = preprocess(df)

    model = build_model((X.shape[1], X.shape[2]))
    history = train_model(model, X, y)

    plot_loss(history)

    # Save model inside folder
    model_path = os.path.join(SAVE_FOLDER, MODEL_NAME)
    model.save(model_path)
    print("\nModel saved to:", model_path)

if __name__ == "__main__":
    main()
