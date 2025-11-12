# ===========================================================
#   PUNTO 1 - Basado en el notebook de ejemplo del profesor
#   Adaptado al dataset aapl.us.txt
# ===========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime
# ===========================================================
# 1. Cargar dataset (CORREGIDO para saltar líneas dañadas)
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

    #Eliminar filas corruptas (NaN)
    df = df.dropna(subset=["Date"] + numeric_cols)

    #Eliminar valores absurdos > 1000 USD (picos falsos del dataset)
    df = df[df["High"] < 1000]
    df = df[df["Low"]  < 1000]

    # ✅ Ordenar por fecha
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ===========================================================
# 2. Crear columna MID — como pide el taller
# ===========================================================

def add_mid_column(df):
    df["Mid"] = (df["High"] + df["Low"]) / 2.0
    return df


# ===========================================================
# 3. Escalar datos — exactamente como el ejemplo usa MinMaxScaler
# ===========================================================

def scale_data(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    return scaled, scaler


# ===========================================================
# 4. Crear ventanas deslizantes — igual que en el ejemplo
# ===========================================================

def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # El ejemplo siempre reshapea X a (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y


# ===========================================================
# 5. Split train/val/test — siguiendo el estilo del ejemplo original
# ===========================================================

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


if __name__ == "__main__":

    print("Cargando datos.")
    df = load_data("data/aapl.us.txt")
    print(df.head())

    print("\nAñadiendo columna MID.")
    df = add_mid_column(df)
    print(df[["Date", "High", "Low", "Mid"]].head())

    # =======================================================
    # Gráfica con información completa
    # =======================================================
    print("\nGraficando MID.")
    os.makedirs("graficas", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_figura = f"graficas/Serie_MID_Apple_{timestamp}.png"
    plt.figure(figsize=(9, 6))
    plt.plot(df["Date"], df["Mid"].values, color="#317EDA", linewidth=1.5, label="Valor promedio (High + Low) / 2")
    plt.title("Evolución del valor promedio diario de la acción AAPL (Apple Inc.)", fontsize=12, fontweight="bold")
    plt.xlabel("Fecha", fontsize=10)
    plt.ylabel("Valor promedio de la acción (USD)", fontsize=10)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(nombre_figura, dpi=300)
    plt.show()

    # =======================================================
    # Escalado y división en conjuntos
    # =======================================================
    print("\n Normalizando datos.")
    mid_scaled, scaler = scale_data(df["Mid"].values)

    print("\n Ventanas de tiempo.")
    N_STEPS = 60
    X, y = create_sliding_windows(mid_scaled, N_STEPS)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")

    print("\nDividiendo en Train / Val / Test.")
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)

    print("\nDivisión completada:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape,   y_val.shape)
    print("Test: ", X_test.shape,  y_test.shape)
