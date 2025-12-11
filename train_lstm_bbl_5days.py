import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib   # for saving the scaler

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
SEQ_LEN = 5                     # use past 5 days
MODEL_NAME = "lstm_bbl_5days.h5"
SCALER_NAME = "bbl_scaler.save"  # joblib file

# ==========================================================
# 1. LOAD DATA
# ==========================================================
def load_data():
    print("Loading dataset:", CSV_FILE)
    df = pd.read_csv(CSV_FILE)

    # --- handle the dataset's "junk" first row (e.g. contains BBL.BK strings)
    # If Date is missing in row 0, drop rows with NaT in Date after conversion.
    # This is safer than blindly dropping the first row.
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).reset_index(drop=True)

    # Ensure numeric columns exist
    expected = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    # Convert numeric columns to float (coerce errors -> NaN) then forward-fill
    df[expected] = df[expected].apply(pd.to_numeric, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    df[expected] = df[expected].fillna(method='ffill').fillna(method='bfill')

    # Keep only needed columns (Date removed for model input)
    df = df[['Date'] + expected]

    print("Data range:", df['Date'].min().date(), "to", df['Date'].max().date())
    print("Rows:", len(df))
    return df

# ==========================================================
# 2. PREPROCESS
# ==========================================================
def preprocess(df):
    print("Scaling OHLCV values...")
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features].values)  # shape (N,5)

    # Save scaler (use joblib for sklearn objects)
    joblib.dump(scaler, SCALER_NAME)
    print("Scaler saved to", SCALER_NAME)

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])   # sequence of shape (SEQ_LEN, 5)
        y.append(scaled[i][3])          # next-day Close is column index 3

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    print("X shape:", X.shape, "y shape:", y.shape)
    return X, y, scaler

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
        Dense(1)  # predict next day close (scaled)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model

# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
def train_model(model, X, y):
    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        shuffle=False,        # don't shuffle time-series sequences
        validation_split=0.1, # small hold-out for validation
        callbacks=[es],
        verbose=1
    )
    return history

# ==========================================================
# 5. PLOT LOSS
# ==========================================================
def plot_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
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

if __name__ == "__main__":
    main()
