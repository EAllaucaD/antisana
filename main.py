import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def cargar_y_preparar_datos(ruta_csv, nombre_columna):
    df = pd.read_csv(ruta_csv)
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
    df = df.set_index('Fecha')

    # Filtrar datos desde 2017
    df = df.loc['2017-01-01':]

    # Seleccionar columna de la estación
    serie = df[nombre_columna].copy()

    # Rellenar valores vacíos con interpolación lineal
    if serie.isnull().sum() > 0:
        serie = serie.interpolate(method='linear')

    # Rellenar valores vacíos restantes con método forward/backward fill
    serie = serie.fillna(method='bfill').fillna(method='ffill')

    return serie

def entrenar_modelo(serie_train):
    modelo = auto_arima(
        serie_train,
        seasonal=True,
        m=3,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    modelo.fit(serie_train)
    return modelo

def evaluar_modelo(test, pred):
    mse = mean_squared_error(test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, pred)
    r2 = r2_score(test, pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def graficar_predicciones(train, test, pred_test, fechas_futuras, pred_futuras):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Entrenamiento')
    plt.plot(test.index, test, label='Prueba')
    plt.plot(test.index, pred_test, label='Predicción prueba')
    plt.plot(fechas_futuras, pred_futuras, label='Predicción futura (36 meses)')
    plt.legend()
    plt.title("Precipitación mensual - Observado y Predicho")
    plt.xlabel("Fecha")
    plt.ylabel("Precipitación")
    plt.grid(True)
    plt.show()

def graficar_barras_futuro(fechas, predicciones):
    plt.figure(figsize=(12,6))
    plt.bar(fechas, predicciones, color='skyblue')
    plt.title("Predicción mensual de precipitación (próximos 3 años)")
    plt.xlabel("Fecha")
    plt.ylabel("Precipitación")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    ruta_csv = 'Precipitacion_Mensual__P55.csv'  # Cambia la ruta si es necesario
    nombre_col = 'P55'

    serie = cargar_y_preparar_datos(ruta_csv, nombre_col)

    # División 70% entrenamiento, 30% prueba
    train_size = int(len(serie) * 0.7)
    train = serie[:train_size]
    test = serie[train_size:]

    modelo = entrenar_modelo(train)

    # Predicción sobre test
    pred_test = modelo.predict(n_periods=len(test))

    metricas = evaluar_modelo(test, pred_test)
    print("Métricas de evaluación (70/30):")
    for k, v in metricas.items():
        print(f"{k}: {v:.4f}")

    # Predicción futura 36 meses (3 años)
    n_futuro = 36
    pred_futuras = modelo.predict(n_periods=n_futuro)
    fechas_futuras = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1), periods=n_futuro, freq='MS')

    graficar_predicciones(train, test, pred_test, fechas_futuras, pred_futuras)
    graficar_barras_futuro(fechas_futuras, pred_futuras)

if __name__ == "__main__":
    main()
