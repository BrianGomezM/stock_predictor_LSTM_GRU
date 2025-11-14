# 🧠 Stock Predictor LSTM-GRU

    Taller 3 - Redes Neuronales | Universidad del Valle (2025-II)
    Predicción del valor promedio de acciones usando modelos LSTM, GRU y redes apiladas.  

## 📘 Descripción general

    Este proyecto implementa modelos de redes neuronales recurrentes (RNN, LSTM, GRU) para predecir el valor promedio diario de una acción 
    bursátil a partir de datos históricos del dataset  https://www.kaggle.com/datasets/borismarjanovic/pricevolume-data-for-all-us-stocks-etfs)


## ⚙️ Requisitos previos

    Python 3.10 o superior  
    pip actualizado  


## 🧩 Instalación y configuración

    1. Crear y activar entorno virtual:
    python3 -3.11 -m venv venv
    .\venv\Scripts\Activate
### MAC
    python3 -m venv venv
    source venv/bin/activate

    2. instalar dependencias:
    pip3 install -r requirements.txt

    3. Verificar que el dataset esté en la carpeta:
    data/aapl.us.txt

    4. Ejecución de los puntos

    Punto 1 – Preprocesamiento y series temporales
    python Punto_1.py

    Punto 2 – Entrenamiento de modelo LSTM
    python Punto_2.py

    Punto 3 – Modelo GRU
    python Punto_3.py

    Punto 4 – Modelos apilados
    python Punto_4.py


