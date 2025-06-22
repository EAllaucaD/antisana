import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from math import sqrt
import warnings
import itertools # Para generar combinaciones de parámetros

# Ignorar advertencias, útil durante el grid search
warnings.filterwarnings("ignore")

def predecir_precipitacion_sarima_con_grid_search(ruta_csv, anios_a_predecir=2, test_size_ratio=0.2):
    """
    Carga datos de precipitación, realiza un grid search para encontrar los mejores
    parámetros SARIMA, entrena el modelo, predice y evalúa el rendimiento.

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

        # *** Manejo de datos faltantes: Interpolación lineal para NaN ***
        df['P55'] = df['P55'].interpolate(method='linear')
        # Rellenar cualquier NaN remanente (ej. al inicio o final de la serie si son NaN puros) con la media
        df['P55'] = df['P55'].fillna(df['P55'].mean())

        # Asegurarse de que la serie sea de frecuencia mensual
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        df = df.reindex(full_date_range) # Reindexar para asegurar todas las fechas mensuales
        df['P55'] = df['P55'].interpolate(method='linear') # Interpolar de nuevo si se crearon NaNs por reindex
        df['P55'] = df['P55'].fillna(df['P55'].mean()) # Última pasada para cualquier NaN


        y = df['P55']
        print(f"\nDatos de precipitación cargados y preparados. Rango de fechas: {y.index.min()} a {y.index.max()}")
        print(f"Cantidad de puntos de datos: {len(y)}")

        # 2. Dividir los datos en conjuntos de entrenamiento y prueba
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

        # 3. Grid Search para encontrar los mejores parámetros SARIMA
        # Puedes ajustar estos rangos si lo consideras necesario.
        # Por ejemplo, p,d,q,P,D,Q = range(0, 3) si quieres probar más combinaciones.
        p = d = q = range(0, 2)  # p, d, q pueden ser 0, 1
        P = D = Q = range(0, 2)  # P, D, Q pueden ser 0, 1
        s = 12                   # Estacionalidad para datos mensuales

        # Generar todas las combinaciones posibles de parámetros (p, d, q)
        pdq = list(itertools.product(p, d, q))
        # Generar todas las combinaciones posibles de parámetros estacionales (P, D, Q, s)
        seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q))]

        best_aic = float("inf")
        best_order = None
        best_seasonal_order = None
        best_results = None

        print(f"\nIniciando Grid Search para SARIMA. Esto puede tomar tiempo...")

        for order in pdq:
            for s_order in seasonal_pdq:
                try:
                    # Entrenar el modelo con cada combinación de parámetros
                    model_gs = sm.tsa.statespace.SARIMAX(train,
                                                         order=order,
                                                         seasonal_order=s_order,
                                                         enforce_stationarity=False,
                                                         enforce_invertibility=False)
                    results_gs = model_gs.fit(disp=False)

                    if results_gs.aic < best_aic:
                        best_aic = results_gs.aic
                        best_order = order
                        best_seasonal_order = s_order
                        best_results = results_gs

                except Exception as e:
                    # print(f"Error con parámetros {order} {s_order}: {e}") # Descomentar para depurar errores de ajuste
                    continue

        if best_order is None:
            raise ValueError("No se pudo encontrar un modelo SARIMA válido con los parámetros probados. Intenta ampliar los rangos.")

        print(f"\nGrid Search completado.")
        print(f"Mejor Modelo SARIMA: AIC={best_aic:.3f}")
        print(f"Mejores Parámetros (p,d,q): {best_order}")
        print(f"Mejores Parámetros Estacionales (P,D,Q,s): {best_seasonal_order}")

        # 4. Usar el mejor modelo encontrado para predicciones y evaluación
        results = best_results
        print("\nSumario del Mejor Modelo SARIMA:")
        print(results.summary())

        # 5. Realizar predicciones sobre el conjunto de prueba
        forecast_test = results.get_forecast(steps=len(test))
        forecast_test_mean = forecast_test.predicted_mean
        forecast_test_ci = forecast_test.conf_int(alpha=0.05)

        print("\nPredicciones generadas para el conjunto de prueba.")

        # 6. Calcular métricas de error
        print("\n--- Métricas de Evaluación en el Conjunto de Prueba ---")
        mse = mean_squared_error(test, forecast_test_mean)
        rmse = sqrt(mse)
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

        # 7. Generar predicciones para los próximos 'anios_a_predecir' años
        num_meses_a_predecir = anios_a_predecir * 12
        last_date_in_data = y.index.max()

        # Generar el rango de fechas para las predicciones futuras
        # Las predicciones se hacen a partir del final del conjunto de entrenamiento
        # Si queremos predecir 24 meses después de la última fecha del dataset original,
        # necesitamos calcular cuántos pasos son desde el final del entrenamiento
        steps_total_forecast = len(test) + num_meses_a_predecir

        forecast_combined = results.get_forecast(steps=steps_total_forecast)
        
        # Separar las predicciones del test y las predicciones futuras
        forecast_future_mean = forecast_combined.predicted_mean.iloc[len(test):]
        forecast_future_ci = forecast_combined.conf_int(alpha=0.05).iloc[len(test):]

        # Asignar el índice correcto a las predicciones futuras
        forecast_future_index = pd.date_range(start=last_date_in_data + pd.DateOffset(months=1),
                                              periods=num_meses_a_predecir,
                                              freq='MS')
        forecast_future_mean.index = forecast_future_index
        forecast_future_ci.index = forecast_future_index


        print(f"\nPredicciones futuras generadas para los próximos {num_meses_a_predecir} meses:")
        print(forecast_future_mean)

        # 8. Guardar las predicciones futuras en un nuevo archivo CSV
        df_predicciones_futuras = pd.DataFrame({
            'Fecha': forecast_future_mean.index,
            'P55_Predicho': forecast_future_mean.values,
            'Lower_CI': forecast_future_ci.iloc[:, 0],
            'Upper_CI': forecast_future_ci.iloc[:, 1]
        })
        nombre_archivo_salida = f"Predicciones_Precipitacion_Futura_{anios_a_predecir}anios.csv"
        df_predicciones_futuras.to_csv(nombre_archivo_salida, index=False)
        print(f"\nPredicciones futuras guardadas en '{nombre_archivo_salida}'")


        # 9. Visualizar resultados
        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train, label='Datos de Entrenamiento', color='blue')
        plt.plot(test.index, test, label='Datos Reales de Prueba', color='green')
        plt.plot(forecast_test_mean.index, forecast_test_mean, label='Predicción en Prueba', color='orange', linestyle='--')
        plt.fill_between(forecast_test_ci.index,
                         forecast_test_ci.iloc[:, 0],
                         forecast_test_ci.iloc[:, 1], color='orange', alpha=0.1, label='IC Predicción Prueba (95%)')

        plt.plot(forecast_future_mean.index, forecast_future_mean, label='Predicción Futura', color='red')
        plt.fill_between(forecast_future_ci.index,
                         forecast_future_ci.iloc[:, 0],
                         forecast_future_ci.iloc[:, 1], color='pink', alpha=0.2, label='IC Predicción Futura (95%)')

        plt.title(f'Precipitación Mensual: Datos, Predicción en Prueba y Predicción Futura ({anios_a_predecir} años)')
        plt.xlabel('Fecha')
        plt.ylabel('Precipitación P55')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_csv}' no se encontró. Asegúrate de que esté en el mismo directorio que el script o proporciona la ruta completa.")
    except ValueError as ve:
        print(f"Error de datos: {ve}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# --- Ejecución del script ---
if __name__ == "__main__":
    nombre_de_tu_csv = 'Precipitacion_Mensual__P55.csv'
    anios_para_predecir = 2 # Predice para los próximos 2 años después de la última fecha del dataset
    proporcion_datos_prueba = 0.2 # 20% de los datos se usarán para evaluar el modelo
    predecir_precipitacion_sarima_con_grid_search(nombre_de_tu_csv, anios_para_predecir, proporcion_datos_prueba)