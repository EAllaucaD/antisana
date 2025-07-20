import matplotlib
matplotlib.use('Agg') 

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
import pmdarima as pm
from flask import Flask, render_template
import os
import markdown 

# Importar la librería de Google Gemini
import google.generativeai as genai

# Importar load_dotenv de python-dotenv
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Cargar variables de entorno desde .env ---
load_dotenv() 

# --- Configuración de la API de Gemini ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

if not GOOGLE_API_KEY:
    print("ADVERTENCIA: La variable de entorno GOOGLE_API_KEY no está configurada.")
    print("Las funcionalidades de Gemini no estarán disponibles. Por favor, asegúrate de que esté en tu archivo .env o configurada manualmente.")

# Configura la API de Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("API de Gemini no configurada debido a la falta de GOOGLE_API_KEY.")


# Configura la ruta de tu archivo CSV y los parámetros de predicción
RUTA_CSV = 'P55_5.csv' # Asegúrate de que este es el nombre correcto de tu archivo CSV mensual
ANIOS_A_PREDECIR = 2 # Esto corresponde a los 24 meses
TEST_SIZE_RATIO = 0.2


# --- FUNCIÓN para obtener recomendaciones de Gemini ---
def obtener_recomendaciones_gemini(future_predictions_df, metrics_data):
    """
    Usa la API de Gemini para interpretar las predicciones de precipitación
    y generar recomendaciones.

    Args:
        future_predictions_df (pd.DataFrame): DataFrame con las predicciones futuras (Fecha, P55_5_Predicho).
        metrics_data (dict): Diccionario con las métricas de evaluación del modelo.

    Returns:
        str: Interpretación y recomendaciones de Gemini, o un mensaje de error.
    """
    if not os.getenv("GOOGLE_API_KEY"): 
        return "No se pudo generar recomendaciones. Clave API de Gemini no configurada en .env."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash') 
        
        # --- MODIFICACIÓN CLAVE DEL PROMPT ---
        prompt_text = "Eres un experto en climatología y gestión de recursos hídricos. "
        prompt_text += "Te proporcionaré predicciones de precipitación mensual para los próximos 24 meses. "
        prompt_text += "Basado **EXCLUSIVAMENTE** en estas predicciones, genera recomendaciones concisas y prácticas. "
        prompt_text += "Ofrece recomendaciones diferenciadas para los siguientes sectores: "
        prompt_text += "**Agricultores**, **Planificadores de Recursos Hídricos** (agua potable, embalses), "
        prompt_text += "y **Gestión de Riesgos** (sequías, inundaciones). "
        prompt_text += "Cada recomendación debe ser un párrafo corto. Formatea tu respuesta usando **Markdown** con encabezados y texto en negrita.\n\n"

        prompt_text += "--- Predicciones de Precipitación Futuras (mm) ---\n"
        if not future_predictions_df.empty:
            # Enviar todas las predicciones para que el modelo tenga el contexto completo
            for index, row in future_predictions_df.iterrows():
                prompt_text += f"Fecha: {row['Fecha']}, Precipitación Predicha: {row['P55_5_Predicho']:.2f} mm\n"
        else:
            prompt_text += "No hay datos de predicciones futuras disponibles para analizar.\n"
            
        prompt_text += "\n--- Contexto de Evaluación del Modelo ---\n"
        for metric, value in metrics_data.items():
            prompt_text += f"{metric}: {value}\n"
        
        prompt_text += "\nPor favor, genera las recomendaciones concisas ahora, organizadas por sector."

        print("\nEnviando solicitud a Gemini...")
        response = model.generate_content(prompt_text)
        print("Respuesta de Gemini recibida.")
        
        return markdown.markdown(response.text)
    except Exception as e:
        print(f"Error al conectar con la API de Gemini: {e}")
        return f"Error al obtener recomendaciones de Gemini: {e}. Asegúrate de que tu clave API sea válida y tengas conexión a internet. Considera probar con 'gemini-1.0-pro' si 'gemini-2.5-flash' no está disponible."


@app.route('/')
def index():
    if not os.path.exists('static'):
        os.makedirs('static')

    metrics, future_predictions_for_html_df, monthly_avg_df = predecir_y_analizar_precipitacion_sarima(
        RUTA_CSV, ANIOS_A_PREDECIR, TEST_SIZE_RATIO
    )

    future_predictions_html = future_predictions_for_html_df.to_html(classes='table table-striped table-bordered', index=False)
    monthly_avg_html = monthly_avg_df.to_frame(name='P55_5_Promedio').to_html(classes='table table-striped table-bordered')

    gemini_recommendations = "Cargando recomendaciones..." 
    if GOOGLE_API_KEY: 
        gemini_recommendations = obtener_recomendaciones_gemini(future_predictions_for_html_df, metrics)
    else:
        gemini_recommendations = "Por favor, configura tu GOOGLE_API_KEY en el archivo .env para obtener recomendaciones de IA."


    return render_template('index.html', 
                            metrics=metrics,
                            future_predictions_html=future_predictions_html,
                            monthly_avg_html=monthly_avg_html, 
                            gemini_recommendations=gemini_recommendations)


def predecir_y_analizar_precipitacion_sarima(ruta_csv, anios_a_predecir=2, test_size_ratio=0.2):
    metrics = {}
    future_predictions_for_html_df = pd.DataFrame() 
    monthly_avg_df = pd.Series()

    try:
        df = pd.read_csv(ruta_csv, sep=';', na_values=[''])
        print(f"DataFrame cargado. Columnas: {df.columns.tolist()}")

        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.set_index('Fecha')
        df = df.sort_index()

        df['P55_5'] = pd.to_numeric(df['P55_5'], errors='coerce') 

        df['P55_5'] = df['P55_5'].interpolate(method='linear')
        df['P55_5'] = df['P55_5'].fillna(df['P55_5'].mean()) 

        # Asegurar que no falte ningún mes en el índice
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        df = df.reindex(full_date_range)
        
        df['P55_5'] = df['P55_5'].interpolate(method='linear')
        df['P55_5'] = df['P55_5'].fillna(df['P55_5'].mean())

        y = df['P55_5']
        print(f"\nDatos de precipitación cargados y preparados. Rango de fechas: {y.index.min()} a {y.index.max()}")
        print(f"Cantidad de puntos de datos: {len(y)}")

        print("\n--- INICIANDO ANÁLISIS DE DATOS Y PREPARACIÓN DE GRÁFICOS ---")

        # --- Descomposición Estacional para verificar patrones ---
        # Para datos mensuales, la estacionalidad es 12
        print("\n--- Realizando Descomposición Estacional ---")
        decomposition = seasonal_decompose(y, model='additive', period=12) 
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle('Descomposición de Serie Temporal (Estacionalidad Mensual)', y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig('static/seasonal_decomposition_P55_5_mensual.png') # Nuevo nombre de archivo
        plt.close()
        print("Gráfico de descomposición estacional guardado como 'static/seasonal_decomposition_P55_5_mensual.png'")

        # Gráficos ACF y PACF para verificar la autocorrelación
        print("\n--- Generando Gráficos ACF y PACF ---")
        plt.figure(figsize=(12, 6))
        plot_acf(y, lags=36, ax=plt.subplot(211)) # 3 * m = 3 * 12 = 36
        plot_pacf(y, lags=36, ax=plt.subplot(212))
        plt.tight_layout()
        plt.savefig('static/acf_pacf_P55_5_mensual.png') # Nuevo nombre de archivo
        plt.close()
        print("Gráficos ACF/PACF guardados como 'static/acf_pacf_P55_5_mensual.png'")


        monthly_avg_df = y.groupby(y.index.month).mean()
        meses_nombres = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                         7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        
        # --- CORRECCIÓN AQUÍ para asegurar que monthly_avg_df tenga todos los meses en orden ---
        # Reindexa para asegurar todos los meses del 1 al 12, rellena si faltan, y luego renombra
        full_months_index = pd.Index(range(1, 13))
        monthly_avg_df = monthly_avg_df.reindex(full_months_index).fillna(0) # Rellena meses sin datos con 0
        monthly_avg_df = monthly_avg_df.rename(index=meses_nombres)
        # -----------------------------------------------------------------------------------

        plt.figure(figsize=(10, 6))
        # Usa el índice de monthly_avg_df (que ahora está correctamente ordenado y renombrado)
        plt.bar(monthly_avg_df.index, monthly_avg_df.values, color='skyblue')
        plt.title('Precipitación Promedio Mensual (Patrón Estacional)')
        plt.xlabel('Mes')
        plt.ylabel('Precipitación Promedio (mm)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('static/precipitacion_mensual_promedio_barras_P55_5.png') # Nombre de archivo actualizado
        plt.close()
        print("Gráfico de barras de precipitación promedio mensual guardado como 'static/precipitacion_mensual_promedio_barras_P55_5.png'")


        print("\n--- INICIANDO MODELADO SARIMA Y PREDICCIONES ---")

        test_size = int(len(y) * test_size_ratio)
        if test_size == 0 and len(y) > 0:
            test_size = 1
        elif len(y) == 0:
            raise ValueError("No hay datos para dividir en entrenamiento y prueba.")

        train = y[:-test_size]
        test = y[-test_size:]

        print(f"\nDividiendo datos: {len(train)} para entrenamiento, {len(test)} para prueba.")
        
        print("\nIniciando búsqueda automática de parámetros SARIMA con auto_arima. Esto puede tomar tiempo...")
        
        stepwise_model = pm.auto_arima(train,
                                      start_p=1, start_q=1, 
                                      test='adf', 
                                      max_p=7, max_q=7, 
                                      m=12, # ESTACIONALIDAD MENSUAL
                                      start_P=0, start_Q=0,
                                      max_P=3, max_Q=3, 
                                      seasonal=True, 
                                      D=1, # Diferenciación estacional (común para precipitación mensual)
                                      d=None, # Permitir que auto_arima determine d, no forzar 1
                                      trace=True, # Cambiar a True para ver el progreso y los modelos probados
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True,
                                      random_state=42, # Para reproducibilidad
                                      n_fits=50, # Número de modelos a probar si stepwise no es exhaustivo
                                      power_transform=True # <-- APLICAR TRANSFORMACIÓN BOX-COX
                                      )

        best_order = stepwise_model.order
        best_seasonal_order = stepwise_model.seasonal_order
        print(f"Parámetros No Estacionales (p,d,q): {best_order}")
        print(f"Parámetros Estacionales (P,D,Q,s): {best_seasonal_order}")

        model = sm.tsa.statespace.SARIMAX(train,
                                          order=best_order,
                                          seasonal_order=best_seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        # Aumentar maxiter para una mejor convergencia del ajuste del modelo
        results = model.fit(disp=False, maxiter=1000) 

        forecast_test = results.get_forecast(steps=len(test))
        forecast_test_mean = forecast_test.predicted_mean
        forecast_test_ci = forecast_test.conf_int(alpha=0.05)

        print("\nPredicciones generadas para el conjunto de prueba.")

        print("\n--- Métricas de Evaluación en el Conjunto de Prueba ---")
        metrics['MSE'] = f"{mean_squared_error(test, forecast_test_mean):.3f}"
        metrics['RMSE'] = f"{np.sqrt(float(metrics['MSE'])):.3f}" 
        metrics['MAE'] = f"{mean_absolute_error(test, forecast_test_mean):.3f}"
        metrics['R2'] = f"{r2_score(test, forecast_test_mean):.3f}"

        if len(test) > 1 and len(forecast_test_mean) > 1:
            correlation, _ = pearsonr(test, forecast_test_mean)
            metrics['Correlación de Pearson'] = f"{correlation:.3f}"
        else:
            metrics['Correlación de Pearson'] = "N/A"
        print(metrics) 

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

        df_para_csv = pd.DataFrame({
            'Fecha': forecast_future_mean.index.strftime('%Y-%m-%d'), 
            'P55_5_Predicho': forecast_future_mean.round(3).values,
            'Lower_CI': forecast_future_ci.iloc[:, 0].round(3).values,
            'Upper_CI': forecast_future_ci.iloc[:, 1].round(3).values
        })
        nombre_archivo_salida = f"Predicciones_Precipitacion_FuturaP55_5_{anios_a_predecir}anios.csv"
        df_para_csv.to_csv(nombre_archivo_salida, index=False)
        print(f"\nPredicciones futuras (completas) guardadas en '{nombre_archivo_salida}'")

        future_predictions_for_html_df = pd.DataFrame({
            'Fecha': forecast_future_mean.index.strftime('%Y-%m-%d'), 
            'P55_5_Predicho': forecast_future_mean.round(3).values
        })

        plt.figure(figsize=(16, 8))
        plt.plot(train.index, train, label='Datos de Entrenamiento (P55_5)', color='blue')
        plt.plot(test.index, test, label='Datos Reales de Prueba (P55_5)', color='green')
        plt.plot(forecast_test_mean.index, forecast_test_mean, label='Predicción en Prueba (P55_5)', color='orange', linestyle='--')
        plt.fill_between(forecast_test_ci.index,
                                     forecast_test_ci.iloc[:, 0],
                                     forecast_test_ci.iloc[:, 1], color='orange', alpha=0.1, label='IC Predicción Prueba (95%)')

        plt.plot(forecast_future_mean.index, forecast_future_mean, label='Predicción Futura (P55_5)', color='red')
        plt.fill_between(forecast_future_ci.index,
                                     forecast_future_ci.iloc[:, 0],
                                     forecast_future_ci.iloc[:, 1], color='pink', alpha=0.2, label='IC Predicción Futura (95%)')

        plt.title(f'Precipitación Mensual: Datos, Predicción en Prueba y Predicción Futura SARIMA ({anios_a_predecir} años)')
        plt.xlabel('Fecha')
        plt.ylabel('Precipitación P55_5 (mm)')
        plt.legend()
        plt.grid(True)
        plt.tick_params(axis='x', rotation=45)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.tight_layout()
        plt.ylim(y.min() - 10, y.max() + 10) 
        plt.savefig('static/prediccion_precipitacion_sarima_P55_5_mensual.png')
        plt.close()
        print("Gráfico de predicciones del modelo SARIMA guardado como 'static/prediccion_precipitacion_sarima_P55_5_mensual.png'")
        
        return metrics, future_predictions_for_html_df, monthly_avg_df

    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_csv}' no se encontró.")
        return {}, pd.DataFrame(), pd.Series()
    except ValueError as ve:
        print(f"Error de datos: {ve}")
        return {}, pd.DataFrame(), pd.Series()
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return {}, pd.DataFrame(), pd.Series()

if __name__ == '__main__':
    app.run(debug=True)