import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.dates import DateFormatter 

# Importar auto_arima
import pmdarima as pm

# Ignorar advertencias, útil durante el grid search
warnings.filterwarnings("ignore")

def predecir_y_analizar_precipitacion_sarima(ruta_csv, anios_a_predecir=2, test_size_ratio=0.2):
    """
    Carga datos de precipitación, usa auto_arima para encontrar los mejores
    parámetros SARIMA, entrena el modelo, predice y evalúa el rendimiento.
    También incluye análisis visual de la serie de tiempo y muestra los gráficos.

    Args:
        ruta_csv (str): Ruta al archivo CSV con los datos de precipitación.
        anios_a_predecir (int): Número de años hacia el futuro para predecir.
        test_size_ratio (float): Proporción de los datos para usar como conjunto de prueba (ej. 0.2 para 20%).
    """
    try:
        # 1. Cargar y preparar los datos
        df = pd.read_csv(ruta_csv)
        print(f"DataFrame cargado. Columnas: {df.columns.tolist()}")

        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.set_index('Fecha')
        df = df.sort_index()

        df['P55'] = pd.to_numeric(df['P55'], errors='coerce') 

        # Manejo de datos faltantes: Interpolación lineal para NaN.
        df['P55'] = df['P55'].interpolate(method='linear')
        df['P55'] = df['P55'].fillna(df['P55'].mean()) # Última pasada para cualquier NaN al principio/final

        # Asegurarse de que la serie sea de frecuencia mensual.
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        df = df.reindex(full_date_range)
        df['P55'] = df['P55'].interpolate(method='linear')
        df['P55'] = df['P55'].fillna(df['P55'].mean())

        y = df['P55']
        print(f"\nDatos de precipitación cargados y preparados. Rango de fechas: {y.index.min()} a {y.index.max()}")
        print(f"Cantidad de puntos de datos: {len(y)}")

        # --- ANÁLISIS Y DATOS PARA CONSOLA Y GRÁFICOS INICIALES ---
        print("\n--- INICIANDO ANÁLISIS DE DATOS Y PREPARACIÓN DE GRÁFICOS ---")

        # Precipitación Promedio Mensual (Datos Numéricos en Consola)
        y_monthly_avg = y.groupby(y.index.month).mean()
        meses_nombres = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                         7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        y_monthly_avg_named = y_monthly_avg.rename(index=meses_nombres)
        print("\n--- Precipitación Promedio Mensual (Datos Numéricos) ---")
        print(y_monthly_avg_named)
        print("---------------------------------------------------------")

        # Gráfico 1: Líneas de la Serie de Tiempo Completa
        plt.figure(figsize=(14, 7)) 
        plt.plot(y.index, y, label='Precipitación P55', color='blue')
        plt.title('Serie de Tiempo de Precipitación Mensual (P55)')
        plt.xlabel('Fecha')
        plt.ylabel('Precipitación P55 (mm)')
        plt.grid(True)
        plt.legend()
        plt.tick_params(axis='x', rotation=45)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.tight_layout()
        plt.ylim(y.min() - 10, y.max() + 10) 
        print("Gráfico de la serie temporal original listo para mostrar.")

        # Gráfico 2: Barras de Precipitación Promedio Mensual (Estacionalidad)
        plt.figure(figsize=(10, 6))
        meses_para_barras = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        plt.bar(meses_para_barras, y_monthly_avg, color='skyblue')
        plt.title('Precipitación Promedio Mensual (Patrón Estacional)')
        plt.xlabel('Mes')
        plt.ylabel('Precipitación Promedio (mm)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        print("Gráfico de barras de precipitación promedio mensual listo para mostrar.")


        # --- MODELADO Y PREDICCIÓN SARIMA ---
        print("\n--- INICIANDO MODELADO SARIMA Y PREDICCIONES ---")

        # Dividir los datos en conjuntos de entrenamiento y prueba
        test_size = int(len(y) * test_size_ratio)
        if test_size == 0 and len(y) > 0:
            test_size = 1
        elif len(y) == 0:
            raise ValueError("No hay datos para dividir en entrenamiento y prueba.")

        train = y[:-test_size]
        test = y[-test_size:]

        print(f"\nDividiendo datos: {len(train)} para entrenamiento, {len(test)} para prueba.")
        print(f"Rango de fechas de entrenamiento: {train.index.min()} a {train.index.max()}")
        print(f"Rango de fechas de prueba: {test.index.min()} a {test.index.max()}")
        
        # Optimización de parámetros con pmdarima.auto_arima
        print("\nIniciando búsqueda automática de parámetros SARIMA con auto_arima. Esto puede tomar tiempo...")
        
        stepwise_model = pm.auto_arima(train,
                                        start_p=0, start_q=0,
                                        test='adf', 
                                        max_p=5, max_q=5, 
                                        m=12, 
                                        start_P=0, start_Q=0,
                                        max_P=2, max_Q=2, 
                                        seasonal=True, 
                                        D=1, 
                                        trace=True, # Mantener en True para ver el progreso durante la búsqueda
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=True)

        print("\nBúsqueda automática de parámetros completada.")
        print("Mejor Modelo SARIMA encontrado por auto_arima:")
        print(stepwise_model.summary())

        best_order = stepwise_model.order
        best_seasonal_order = stepwise_model.seasonal_order
        print(f"Parámetros No Estacionales (p,d,q): {best_order}")
        print(f"Parámetros Estacionales (P,D,Q,s): {best_seasonal_order}")

        # Usar el mejor modelo encontrado por auto_arima para predicciones y evaluación
        model = sm.tsa.statespace.SARIMAX(train,
                                            order=best_order,
                                            seasonal_order=best_seasonal_order,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
        results = model.fit(disp=False)
        print("\nSumario del Modelo SARIMA final:")
        print(results.summary())

        # Realizar predicciones sobre el conjunto de prueba
        forecast_test = results.get_forecast(steps=len(test))
        forecast_test_mean = forecast_test.predicted_mean
        forecast_test_ci = forecast_test.conf_int(alpha=0.05)

        print("\nPredicciones generadas para el conjunto de prueba.")

        # Calcular métricas de error
        print("\n--- Métricas de Evaluación en el Conjunto de Prueba ---")
        mse = mean_squared_error(test, forecast_test_mean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, forecast_test_mean)
        r2 = r2_score(test, forecast_test_mean)

        if len(test) > 1 and len(forecast_test_mean) > 1:
            correlation, _ = pearsonr(test, forecast_test_mean)
        else:
            correlation = np.nan

        print(f"MSE (Error Cuadrático Medio): {mse:.3f}")
        print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.3f}")
        print(f"MAE (Error Absoluto Medio): {mae:.3f}")
        print(f"R² (Coeficiente de Determinación): {r2:.3f}")
        print(f"Correlación de Pearson: {correlation:.3f}")
        print("--------------------------------------------------")

        # Generar predicciones para los próximos 'anios_a_predecir' años
        num_meses_a_predecir = anios_a_predecir * 12
        last_date_in_data = y.index.max()

        steps_total_forecast = len(test) + num_meses_a_predecir

        forecast_combined = results.get_forecast(steps=steps_total_forecast)
        
        forecast_future_mean = forecast_combined.predicted_mean.iloc[len(test):]
        forecast_future_ci = forecast_combined.conf_int(alpha=0.05).iloc[len(test):]

        forecast_future_index = pd.date_range(start=last_date_in_data + pd.DateOffset(months=1),
                                                periods=num_meses_a_predecir,
                                                freq='MS')
        forecast_future_mean.index = forecast_future_index
        forecast_future_ci.index = forecast_future_index

        print(f"\nPredicciones futuras generadas para los próximos {num_meses_a_predecir} meses:")
        print(forecast_future_mean)

        # Guardar las predicciones futuras en un nuevo archivo CSV
        df_predicciones_futuras = pd.DataFrame({
            'Fecha': forecast_future_mean.index,
            'P55_Predicho': forecast_future_mean.values,
            'Lower_CI': forecast_future_ci.iloc[:, 0],
            'Upper_CI': forecast_future_ci.iloc[:, 1]
        })
        nombre_archivo_salida = f"Predicciones_Precipitacion_Futura_{anios_a_predecir}anios.csv"
        df_predicciones_futuras.to_csv(nombre_archivo_salida, index=False)
        print(f"\nPredicciones futuras guardadas en '{nombre_archivo_salida}'")


        # --- GRÁFICO FINAL DE PREDICCIONES ---
        plt.figure(figsize=(16, 8))
        plt.plot(train.index, train, label='Datos de Entrenamiento (P55)', color='blue')
        plt.plot(test.index, test, label='Datos Reales de Prueba (P55)', color='green')
        plt.plot(forecast_test_mean.index, forecast_test_mean, label='Predicción en Prueba (P55)', color='orange', linestyle='--')
        plt.fill_between(forecast_test_ci.index,
                            forecast_test_ci.iloc[:, 0],
                            forecast_test_ci.iloc[:, 1], color='orange', alpha=0.1, label='IC Predicción Prueba (95%)')

        plt.plot(forecast_future_mean.index, forecast_future_mean, label='Predicción Futura (P55)', color='red')
        plt.fill_between(forecast_future_ci.index,
                            forecast_future_ci.iloc[:, 0],
                            forecast_future_ci.iloc[:, 1], color='pink', alpha=0.2, label='IC Predicción Futura (95%)')

        plt.title(f'Precipitación Mensual: Datos, Predicción en Prueba y Predicción Futura SARIMA ({anios_a_predecir} años)')
        plt.xlabel('Fecha')
        plt.ylabel('Precipitación P55 (mm)')
        plt.legend()
        plt.grid(True)
        plt.tick_params(axis='x', rotation=45)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.tight_layout()
        plt.ylim(y.min() - 10, y.max() + 10) 
        print("Gráfico de predicciones del modelo SARIMA listo para mostrar.")

        # --- Mostrar todos los gráficos generados simultáneamente al final ---
        plt.show() 
        print("\nTodos los gráficos solicitados han sido mostrados.")

    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_csv}' no se encontró. Asegúrate de que esté en el mismo directorio que el script o proporciona la ruta completa.")
    except ValueError as ve:
        print(f"Error de datos: {ve}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# --- Ejecución del script ---
if __name__ == "__main__":
    nombre_de_tu_csv = 'Precipitacion_Mensual__P55.csv'
    anios_para_predecir = 2
    proporcion_datos_prueba = 0.2
    predecir_y_analizar_precipitacion_sarima(nombre_de_tu_csv, anios_para_predecir, proporcion_datos_prueba)