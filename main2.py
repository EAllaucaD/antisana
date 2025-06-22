import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Funciones ---
def cargar_y_preparar_datos(ruta_csv, columna_fecha, columna_valor):
    df = pd.read_csv(ruta_csv)
    
    # Convertir a datetime sin formato explícito (detecta automáticamente yyyy-mm-dd)
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    
    # Rellenar valores vacíos con promedio de valores anterior y posterior
    serie = df[columna_valor].copy()
    for i in range(len(serie)):
        if pd.isna(serie.iloc[i]):
            prev_val = serie.iloc[i-1] if i-1 >= 0 else np.nan
            next_val = serie.iloc[i+1] if i+1 < len(serie) else np.nan
            
            if not np.isnan(prev_val) and not np.isnan(next_val):
                serie.iloc[i] = (prev_val + next_val) / 2
    
    # Interpolación lineal para múltiples NaNs consecutivos o extremos
    if serie.isnull().sum() > 0:
        serie = serie.interpolate(method='linear')
    serie = serie.fillna(method='bfill').fillna(method='ffill')
    
    df[columna_valor] = serie
    
    df.set_index(columna_fecha, inplace=True)
    df.index.freq = 'MS'  # mensual
    
    return df[columna_valor]

def separar_datos(serie, porcentaje_train=0.7):
    train_size = int(len(serie) * porcentaje_train)
    train = serie.iloc[:train_size]
    test = serie.iloc[train_size:]
    return train, test

def sarima_forecast(train, test, order, seasonal_order, n_periods):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(test) + n_periods)
    return forecast

def print_errors(test, forecast):
    mse = mean_squared_error(test, forecast[:len(test)])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, forecast[:len(test)])
    r2 = r2_score(test, forecast[:len(test)])
    corr = np.corrcoef(test, forecast[:len(test)])[0,1]
    
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'Correlación: {corr:.4f}')

def graficar_resultados(train, test, forecast, n_periods):
    plt.figure(figsize=(14,6))
    plt.plot(train.index, train, label='Entrenamiento')
    plt.plot(test.index, test, label='Prueba')
    # Crear rango fechas para forecast (test + n_periods)
    forecast_index = pd.date_range(start=test.index[0], periods=len(test)+n_periods, freq='M')
    plt.plot(forecast_index, forecast, label='Pronóstico SARIMA', color='red')
    plt.title('Predicción SARIMA para Precipitación')
    plt.xlabel('Fecha')
    plt.ylabel('Precipitación')
    plt.legend()
    plt.show()

# --- Main ---
def main():
    ruta_csv = 'Precipitacion_Mensual__P55.csv'  # Ajusta al nombre y ruta real
    columna_fecha = 'Fecha'  # Cambia al nombre real de tu columna fecha
    columna_valor = 'P55'  # Cambia al nombre real de tu columna de datos
    
    n_periods = 12  # Número de meses a predecir
    
    # Cargar y preparar datos
    serie = cargar_y_preparar_datos(ruta_csv, columna_fecha, columna_valor)
    
    # Separar datos
    train, test = separar_datos(serie)
    
    # Orden SARIMA ejemplo
    order = (1,1,1)
    seasonal_order = (1,1,1,12)  # Mensual con estacionalidad anual
    
    # Predecir con SARIMA
    forecast = sarima_forecast(train, test, order, seasonal_order, n_periods)
    
    # Mostrar errores
    print_errors(test, forecast)
    
    # Graficar resultados
    graficar_resultados(train, test, forecast, n_periods)
    
    # Mostrar predicciones futuras
    print(f'\nPredicciones para las próximas {n_periods} periodos:')
    for i, val in enumerate(forecast[-n_periods:]):
        print(f'Periodo {i+1}: {val:.2f}')

if __name__ == '__main__':
    main()
