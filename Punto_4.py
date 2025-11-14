# ===========================================================
#   PUNTO 4 - Red recurrente apilada (Stacked LSTM)
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.optimizers import Adam

# ===========================================================
#   1. Cargar datos (del Punto 1)
# ===========================================================

def load_data(path):
    df = pd.read_csv(
        path,
        header=None,
        names=["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"],
        on_bad_lines='skip',
        engine='python'
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "OpenInt"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df[df["High"] < 1000]
    df = df[df["Low"] < 1000]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_mid_column(df):
    df["Mid"] = (df["High"] + df["Low"]) / 2.0
    return df


def scale_data(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    return scaled, scaler


def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y


def split_train_val_test(X, y):
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test


# ===========================================================
#   2. Preparación del dataset
# ===========================================================

df = load_data("data/aapl.us.txt")
df = add_mid_column(df)
mid_scaled, scaler = scale_data(df["Mid"].values)

N_STEPS = 60
X, y = create_sliding_windows(mid_scaled, N_STEPS)

X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)

print("\nDataset preparado")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


# ===========================================================
#   3. Construcción del modelo LSTM apilado
# ===========================================================

def build_stacked_lstm(neurons=64, dropout=False):
    model = Sequential()

    # Primera capa LSTM
    model.add(LSTM(
        neurons,
        return_sequences=True,
        recurrent_dropout=0.2 if dropout else 0.0,
        input_shape=(X_train.shape[1], 1)
    ))

    # Segunda capa LSTM apilada
    model.add(LSTM(
        neurons,
        return_sequences=False,
        recurrent_dropout=0.2 if dropout else 0.0
    ))

    # Capa final densa
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss="mse"
    )

    return model


# ===========================================================
#   4. Entrenamiento
# ===========================================================

print("\n===== ENTRENANDO MODELO APILADO (STACKED LSTM) =====")

config = {
    "neurons": 64,
    "epochs": 20,
    "batch": 64,
    "dropout": False
}

print("Configuración utilizada:", config)

model = build_stacked_lstm(
    neurons=config["neurons"],
    dropout=config["dropout"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["epochs"],
    batch_size=config["batch"],
    verbose=1
)

# ===========================================================
#   5. Gráficas
# ===========================================================

def plot_loss(history):
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Curva de pérdida - Stacked LSTM")
    plt.xlabel("Iteraciones")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_predictions(y_real, y_pred):
    plt.figure(figsize=(10,5))
    plt.plot(y_real, label="Real", linewidth=2)
    plt.plot(y_pred, label="Predicción (Stacked LSTM)", linewidth=2)
    plt.title("Predicción vs Real - Modelo LSTM Apilado")
    plt.xlabel("Tiempo")
    plt.ylabel("MID Normalizado")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


plot_loss(history)

y_pred = model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

plot_predictions(y_test, y_pred)


# ===========================================================
#   6. Resultados finales
# ===========================================================

print("\n==============================")
print("  RESULTADOS DEL MODELO APILADO")
print("==============================")
print(f"RMSE final: {rmse}")
