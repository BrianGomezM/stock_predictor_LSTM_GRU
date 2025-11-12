# ===========================================================
#   PUNTO 4 - Red Apilada (Stacked LSTM / GRU)
#   Combinación de dos capas recurrentes para mejorar desempeño
#   Basado en los flujos de Puntos 2 (LSTM) y 3 (GRU)
# ===========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ------------------------------
# 1) Utilidades del Punto 1
# ------------------------------

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
    df = df.dropna(subset=["Date"] + numeric_cols)
    df = df[df["High"] < 1000]
    df = df[df["Low"]  < 1000]
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
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def split_train_val_test(X, y):
    train_size = int(len(X) * 0.70)
    val_size   = int(len(X) * 0.15)
    X_train = X[:train_size];             y_train = y[:train_size]
    X_val   = X[train_size:train_size+val_size]; y_val   = y[train_size:train_size+val_size]
    X_test  = X[train_size+val_size:];    y_test  = y[train_size+val_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test

# ------------------------------
# 2) Métricas
# ------------------------------

def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
def mape(y_true, y_pred, eps=1e-8): return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0

# ------------------------------
# 3) Modelo Apilado (Stacked)
# ------------------------------

def build_stacked_model(input_shape, units_1=64, units_2=32, dropout=0.2, mode="LSTM-GRU"):
    model = keras.Sequential()
    if mode == "LSTM-LSTM":
        model.add(layers.LSTM(units_1, return_sequences=True, recurrent_dropout=dropout, input_shape=input_shape))
        model.add(layers.LSTM(units_2, recurrent_dropout=dropout))
    elif mode == "LSTM-GRU":
        model.add(layers.LSTM(units_1, return_sequences=True, recurrent_dropout=dropout, input_shape=input_shape))
        model.add(layers.GRU(units_2, recurrent_dropout=dropout))
    elif mode == "GRU-GRU":
        model.add(layers.GRU(units_1, return_sequences=True, recurrent_dropout=dropout, input_shape=input_shape))
        model.add(layers.GRU(units_2, recurrent_dropout=dropout))
    else:
        raise ValueError("Modo no soportado. Usa: 'LSTM-LSTM', 'LSTM-GRU', 'GRU-GRU'.")

    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

# ------------------------------
# 4) Entrenamiento y Evaluación
# ------------------------------

def train_and_evaluate(config, X_train, y_train, X_val, y_val, X_test, y_test, scaler, exp_id):
    u1, u2, ep, bs, dp, mode = config.values()

    model = build_stacked_model(X_train.shape[1:], units_1=u1, units_2=u2, dropout=dp, mode=mode)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ckpt = f"stacked_best_exp{exp_id}.keras"
    mc = callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=ep,
        batch_size=bs,
        callbacks=[es, mc],
        verbose=0
    )

    model = keras.models.load_model(ckpt)
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_scaled = y_test.reshape(-1, 1)

    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler.inverse_transform(y_test_scaled).ravel()

    _mae, _rmse, _mape = mae(y_true, y_pred), rmse(y_true, y_pred), mape(y_true, y_pred)

    os.makedirs("resultados_p4", exist_ok=True)

    # Gráfica de pérdida
    plt.figure(figsize=(7,5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title(f"Pérdida - {mode} (u1={u1}, u2={u2}, ep={ep}, bs={bs})")
    plt.xlabel("Épocas"); plt.ylabel("MSE")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    loss_fig = f"resultados_p4/exp{exp_id}_loss.png"
    plt.savefig(loss_fig, dpi=300); plt.close()

    # Gráfica Predicción vs Real
    N = min(200, len(y_true))
    plt.figure(figsize=(9,5))
    plt.plot(range(N), y_true[-N:], label="Real")
    plt.plot(range(N), y_pred[-N:], label="Predicción")
    plt.title(f"Predicción vs Real - {mode} (u1={u1}, u2={u2}, ep={ep})")
    plt.xlabel("Tiempo"); plt.ylabel("Precio promedio (USD)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    pred_fig = f"resultados_p4/exp{exp_id}_pred.png"
    plt.savefig(pred_fig, dpi=300); plt.close()

    return {"exp_id": exp_id, "mode": mode, "u1": u1, "u2": u2, "ep": ep, "bs": bs, "dp": dp,
            "MAE": _mae, "RMSE": _rmse, "MAPE(%)": _mape,
            "loss_fig": loss_fig, "pred_fig": pred_fig, "ckpt": ckpt}

# ------------------------------
# 5) Main
# ------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    df = add_mid_column(load_data("data/aapl.us.txt"))
    mid_scaled, scaler = scale_data(df["Mid"].values)
    X, y = create_sliding_windows(mid_scaled, 60)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)

    # Configuraciones
    configs = [
        {"u1": 64, "u2": 32, "ep": 30, "bs": 32, "dp": 0.2, "mode": "LSTM-LSTM"},
        {"u1": 64, "u2": 32, "ep": 30, "bs": 32, "dp": 0.2, "mode": "LSTM-GRU"},
        {"u1": 32, "u2": 16, "ep": 40, "bs": 64, "dp": 0.3, "mode": "GRU-GRU"},
    ]

    resultados = []
    for i, cfg in enumerate(configs, start=1):
        print(f"\n=== Experimento {i}/{len(configs)}: {cfg['mode']} ===")
        res = train_and_evaluate(cfg, X_train, y_train, X_val, y_val, X_test, y_test, scaler, exp_id=i)
        print(f"-> MAE: {res['MAE']:.4f} | RMSE: {res['RMSE']:.4f} | MAPE: {res['MAPE(%)']:.2f}%")
        resultados.append(res)

    df_res = pd.DataFrame(resultados).sort_values(by="RMSE", ascending=True)
    df_res.to_csv("resultados_p4/ranking_resultados_p4.csv", index=False)
    print("\n===== Ranking final (Stacked) =====")
    print(df_res[["exp_id","mode","u1","u2","ep","bs","dp","MAE","RMSE","MAPE(%)"]])
