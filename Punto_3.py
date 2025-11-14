# ===========================================================
#   PUNTO 3 - Modelo GRU para predecir MID (t+1)
# ===========================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error
import math
import os
from datetime import datetime
from Punto_1 import load_data, add_mid_column, scale_data, create_sliding_windows, split_train_val_test
import pandas as pd
from tensorflow.keras.optimizers import Adam

df = load_data("data/aapl.us.txt")
df = add_mid_column(df)

# Datos necesarios
mid_scaled, scaler = scale_data(df["Mid"].values)

N_STEPS = 60
X, y = create_sliding_windows(mid_scaled, N_STEPS)

X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)

# ===========================================================
# 1. Función para crear el modelo GRU
# ===========================================================

def build_gru_model(neurons=50, dropout=False):
    model = Sequential()
    model.add(GRU(
        neurons,
        return_sequences=False,
        recurrent_dropout=0.2 if dropout else 0.0,
        input_shape=(X_train.shape[1], 1)
    ))
    model.add(Dense(1))

    model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="mse"
)
    return model


# ===========================================================
# 2. Función para graficar la pérdida
# ===========================================================

def plot_loss(history, title):
    plt.figure(figsize=(7,5))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title(title)
    plt.xlabel("Iteraciones")
    plt.ylabel("Pérdida (MSE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ===========================================================
# 3. Función para graficar predicción vs real
# ===========================================================

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8,5))
    plt.plot(y_true, label="Real", linewidth=2)
    plt.plot(y_pred, label="Predicción GRU", linewidth=2)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Valor MID (normalizado)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ===========================================================
# 4. Entrenar modelos con 3 configuraciones distintas
# ===========================================================

configs = [
    {"neurons": 32, "epochs": 20, "batch": 64, "dropout": False},
    {"neurons": 64, "epochs": 15, "batch": 32, "dropout": False},
    {"neurons": 32, "epochs": 5, "batch": 64, "dropout": True},
]

results = []
best_rmse = 9999
best_model = None
best_config = None

print("\n===== ENTRENANDO MODELOS GRU =====\n")

for i, cfg in enumerate(configs):
    print(f"\n➤ CONFIGURACIÓN {i+1}: {cfg}")

    model = build_gru_model(cfg["neurons"], cfg["dropout"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch"],
        verbose=1
    )

    # Graficar pérdidas
    plot_loss(history, f"Pérdida - GRU Config {i+1}")

    # Predicciones sobre test
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    print(f"RMSE Config {i+1}: {rmse}")

    # Guardar resultado
    results.append((cfg, rmse))

    # ¿Es el mejor modelo?
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_config = cfg

    # Graficar pred vs real
    plot_predictions(y_test, y_pred, f"Predicción GRU - Config {i+1}")


# ===========================================================
# 5. Mostrar la mejor configuración
# ===========================================================

print("\n==============================")
print(" MEJOR CONFIGURACIÓN DEL GRU")
print("==============================")
print(best_config)
print(f"RMSE = {best_rmse}")

