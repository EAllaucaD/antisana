import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.dates import DateFormatter # Importar para formato de fecha

# Importar auto_arima
import pmdarima as pm

# Ignorar advertencias, útil durante el grid search
warnings.filterwarnings("ignore")

def predecir_precipitacion_sarima_auto_arima_y_analisis_visual(ruta_csv, anios_a_predecir=2, test_size_ratio=0.2):
    """
    Carga datos de precipitación, usa auto_arima para encontrar los mejores
    parámetros SARIMA, entrena el modelo, predice y evalúa el rendimiento.
    También incluye análisis visual de la serie de tiempo.

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

        df['P55'] = pd.to_numeric(df['P55'], errors='coerce') # Usar 'coerce' para convertir a NaN si hay errores

        # Manejo de datos faltantes: Interpolación lineal para NaN.
        # Esto es crucial ya que auto_arima (y SARIMA) esperan una serie completa.
        df['P55'] = df['P55'].interpolate(method='linear')
        df['P55'] = df['P55'].fillna(df['P55'].mean()) # Última pasada para cualquier NaN al principio/final

        # Asegurarse de que la serie sea de frecuencia mensual.
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        df = df.reindex(full_date_range)
        # Después de reindexar, aplicar interpolación y relleno nuevamente para cualquier nuevo NaN.
        df['P55'] = df['P55'].interpolate(method='linear')
        df['P55'] = df['P55'].fillna(df['P55'].mean())


        y = df['P55']
        print(f"\nDatos de precipitación cargados y preparados. Rango de fechas: {y.index.min()} a {y.index.max()}")
        print(f"Cantidad de puntos de datos: {len(y)}")

        # --- Análisis Visual de la Serie de Tiempo ---
        print("\n--- Realizando Análisis Visual de la Serie de Tiempo ---")

        # Gráfico de Líneas de la Serie de Tiempo Completa
        plt.figure(figsize=(14, 7)) # Aumentar tamaño para mejor visibilidad
        plt.plot(y.index, y, label='Precipitación P55', color='blue')
        plt.title('Serie de Tiempo de Precipitación Mensual (P55)')
        plt.xlabel('Fecha')
        plt.ylabel('Precipitación P55 (mm)') # Etiqueta más descriptiva
        plt.grid(True)
        plt.legend()
        plt.tick_params(axis='x', rotation=45) # Rotar etiquetas del eje X
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m')) # Formato de fecha Año-Mes
        plt.tight_layout()
        plt.savefig('serie_temporal_precipitacion.png')
        plt.close()
        print("Gráfico de la serie temporal original guardado como 'serie_temporal_precipitacion.png'")

        # Nuevo Gráfico de Barras: Precipitación Promedio Mensual (Estacionalidad)
        plt.figure(figsize=(10, 6))
        # Calcular la precipitación promedio por mes del año
        y_monthly_avg = y.groupby(y.index.month).mean()
        # Mapear los números de mes a nombres para mejor legibilidad
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        plt.bar(meses, y_monthly_avg, color='skyblue')
        plt.title('Precipitación Promedio Mensual (Patrón Estacional)')
        plt.xlabel('Mes')
        plt.ylabel('Precipitación Promedio (mm)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('precipitacion_mensual_promedio_barras.png')
        plt.close()
        print("Gráfico de barras de precipitación promedio mensual guardado como 'precipitacion_mensual_promedio_barras.png'")


        try:
            decomposicion = seasonal_decompose(y, model='additive', period=12)
            # Aumentar el tamaño de la figura para la descomposición
            fig = decomposicion.plot()
            fig.set_size_inches(12, 10) # Ajustar el tamaño
            plt.suptitle('Descomposición de la Serie de Precipitación', y=1.02)
            plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Ajustar layout para título
            plt.savefig('descomposicion_serie_precipitacion.png')
            plt.close()
            print("Gráfico de descomposición de la serie guardado como 'descomposicion_serie_precipitacion.png'")
        except Exception as e:
            print(f"No se pudo generar el gráfico de descomposición: {e}")

        lags_to_use = min(40, int(len(y) / 2) - 1)

        plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(211)
        plot_acf(y, lags=lags_to_use, ax=ax1, title=f'Función de Autocorrelación (ACF) de la Precipitación (Lags={lags_to_use})')
        plt.ylabel('Coeficiente de Autocorrelación') # Etiqueta para el eje Y
        ax2 = plt.subplot(212)
        plot_pacf(y, lags=lags_to_use, ax=ax2, title=f'Función de Autocorrelación Parcial (PACF) de la Precipitación (Lags={lags_to_use})')
        plt.ylabel('Coeficiente de Autocorrelación Parcial') # Etiqueta para el eje Y
        plt.tight_layout()
        plt.savefig('acf_pacf_precipitacion.png')
        plt.close()
        print(f"Gráficos de ACF y PACF guardados como 'acf_pacf_precipitacion.png' (Lags utilizados: {lags_to_use})")

        y_diff = y.diff().dropna()
        if len(y_diff) > 1:
            lags_diff_to_use = min(40, int(len(y_diff) / 2) - 1)
            plt.figure(figsize=(14, 8))
            ax3 = plt.subplot(211)
            plot_acf(y_diff, lags=lags_diff_to_use, ax=ax3, title=f'ACF de la 1ª Diferencia de la Precipitación (Lags={lags_diff_to_use})')
            plt.ylabel('Coeficiente de Autocorrelación')
            ax4 = plt.subplot(212)
            plot_pacf(y_diff, lags=lags_diff_to_use, ax=ax4, title=f'PACF de la 1ª Diferencia de la Precipitación (Lags={lags_diff_to_use})')
            plt.ylabel('Coeficiente de Autocorrelación Parcial')
            plt.tight_layout()
            plt.savefig('acf_pacf_1ra_diferencia_precipitacion.png')
            plt.close()
            print(f"Gráficos de ACF y PACF de la 1ª diferencia guardados como 'acf_pacf_1ra_diferencia_precipitacion.png' (Lags utilizados: {lags_diff_to_use})")
        else:
            print("No hay suficientes datos para generar gráficos ACF/PACF de la primera diferencia.")

        print("--- Análisis Visual Completado ---")

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

        # --- Diagnóstico de los datos de entrenamiento antes de auto_arima ---
        print("\n--- Diagnóstico de los datos de entrenamiento antes de auto_arima ---")
        print(f"Valores nulos en 'train' (P55): {train.isnull().sum()}")
        print(f"Descripción de 'train' (P55):\n{train.describe()}")
        
        # Visualizar la serie de entrenamiento para una inspección rápida
        plt.figure(figsize=(10, 4))
        plt.plot(train.index, train, label='Serie de Entrenamiento (P55)')
        plt.title('Serie de Entrenamiento de Precipitación')
        plt.xlabel('Fecha')
        plt.ylabel('Precipitación P55 (mm)') # Etiqueta más descriptiva
        plt.grid(True)
        plt.legend()
        plt.tick_params(axis='x', rotation=45) # Rotar etiquetas del eje X
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m')) # Formato de fecha Año-Mes
        plt.tight_layout()
        plt.savefig('train_serie_precipitacion.png')
        plt.close()
        print("Gráfico de la serie de entrenamiento guardado como 'train_serie_precipitacion.png'")

        print("--- Fin del Diagnóstico de los datos de entrenamiento ---")


        # 3. Optimización de parámetros con pmdarima.auto_arima
        print("\nIniciando búsqueda automática de parámetros SARIMA con auto_arima. Esto puede tomar tiempo...")
        # key parameters for auto_arima:
        # seasonal=True: Enable seasonal components
        # m=12: Seasonal period (12 for monthly data)
        # start_p, start_q, start_P, start_Q: Starting values for orders
        # max_p, max_q, max_P, max_Q: Maximum orders to test
        # D: Seasonal differencing order
        # stepwise=True: Use stepwise algorithm for faster search
        # suppress_warnings=True: Suppress auto_arima warnings
        # trace=True: Show progress in console

        # Ajustamos los rangos a los que auto_arima buscará.
        # Puedes ajustarlos si el análisis visual (ACF/PACF) sugiere rangos más amplios/reducidos.
        # Cuidado con max_p/q/P/Q, valores altos aumentan el tiempo de ejecución exponencialmente.
        # Aquí, estamos siendo un poco más agresivos que el grid search manual anterior.
        
        stepwise_model = pm.auto_arima(train,
                                       start_p=0, start_q=0,
                                       test='adf', # Usar test de Dickey-Fuller aumentado para determinar d
                                       max_p=5, max_q=5, # Máximo orden AR/MA no estacional
                                       m=12, # Frecuencia estacional para datos mensuales
                                       start_P=0, start_Q=0,
                                       max_P=2, max_Q=2, # Máximo orden AR/MA estacional
                                       seasonal=True, # Habilitar componentes estacionales
                                       D=1, # Intentar con diferencia estacional de orden 1
                                       trace=True,
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


        # 4. Usar el mejor modelo encontrado por auto_arima para predicciones y evaluación
        # Volvemos a ajustar el modelo con statsmodels para usar sus métodos de forecast.
        model = sm.tsa.statespace.SARIMAX(train,
                                           order=best_order,
                                           seasonal_order=best_seasonal_order,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False)
        results = model.fit(disp=False)
        print("\nSumario del Modelo SARIMA final:")
        print(results.summary())

        # 5. Realizar predicciones sobre el conjunto de prueba
        forecast_test = results.get_forecast(steps=len(test))
        forecast_test_mean = forecast_test.predicted_mean
        forecast_test_ci = forecast_test.conf_int(alpha=0.05)

        print("\nPredicciones generadas para el conjunto de prueba.")

        # 6. Calcular métricas de error
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

        # 7. Generar predicciones para los próximos 'anios_a_predecir' años
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


        # 9. Visualizar resultados del modelo
        plt.figure(figsize=(16, 8)) # Aumentar el tamaño para más espacio
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
        plt.ylabel('Precipitación P55 (mm)') # Etiqueta más descriptiva
        plt.legend()
        plt.grid(True)
        plt.tick_params(axis='x', rotation=45) # Rotar etiquetas del eje X
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m')) # Formato de fecha Año-Mes
        plt.tight_layout()
        plt.savefig('prediccion_precipitacion_sarima.png')
        plt.close()
        print("Gráfico de predicciones del modelo SARIMA guardado como 'prediccion_precipitacion_sarima.png'")


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
    predecir_precipitacion_sarima_auto_arima_y_analisis_visual(nombre_de_tu_csv, anios_para_predecir, proporcion_datos_prueba)