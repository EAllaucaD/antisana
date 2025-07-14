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


# --- Parámetros globales para ambas estaciones ---
ANIOS_A_PREDECIR = 2 # Esto corresponde a los 24 meses
TEST_SIZE_RATIO = 0.2

# --- FUNCIÓN para obtener recomendaciones de Gemini ---
def obtener_recomendaciones_gemini(future_predictions_df, metrics_data, estacion_nombre):
    """
    Usa la API de Gemini para interpretar las predicciones de precipitación
    y generar recomendaciones.

    Args:
        future_predictions_df (pd.DataFrame): DataFrame con las predicciones futuras (Fecha, PXX_Predicho).
        metrics_data (dict): Diccionario con las métricas de evaluación del modelo.
        estacion_nombre (str): Nombre de la columna de la estación (e.g., 'P55', 'P43').

    Returns:
        str: Interpretación y recomendaciones de Gemini, o un mensaje de error.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        return "No se pudo generar recomendaciones. Clave API de Gemini no configurada en .env."

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        # --- MODIFICACIÓN CLAVE DEL PROMPT ---
        prompt_text = "Eres un experto en climatología y gestión de recursos hídricos. "
        prompt_text += f"Te proporcionaré predicciones de precipitación mensual para la estación **{estacion_nombre}** para los próximos 24 meses. "
        prompt_text += "Basado **EXCLUSIVAMENTE** en estas predicciones, genera recomendaciones concisas y prácticas. "
        prompt_text += "Ofrece recomendaciones diferenciadas para los siguientes sectores: "
        prompt_text += "**Agricultores**, **Planificadores de Recursos Hídricos** (agua potable, embalses), "
        prompt_text += "y **Gestión de Riesgos** (sequías, inundaciones). "
        prompt_text += "Cada recomendación debe ser un párrafo corto. Formatea tu respuesta usando **Markdown** con encabezados y texto en negrita.\n\n"

        prompt_text += f"--- Predicciones de Precipitación Futuras (mm) para {estacion_nombre} ---\n"
        pred_col_name = f'{estacion_nombre}_Predicho'
        if not future_predictions_df.empty:
            for index, row in future_predictions_df.iterrows():
                prompt_text += f"Fecha: {row['Fecha']}, Precipitación Predicha: {row[pred_col_name]:.2f} mm\n"
        else:
            prompt_text += "No hay datos de predicciones futuras disponibles para analizar.\n"

        prompt_text += "\n--- Contexto de Evaluación del Modelo ---\n"
        for metric, value in metrics_data.items():
            prompt_text += f"{metric}: {value}\n"

        prompt_text += "\nPor favor, genera las recomendaciones concisas ahora, organizadas por sector."

        print(f"\nEnviando solicitud a Gemini para {estacion_nombre}...")
        response = model.generate_content(prompt_text)
        print(f"Respuesta de Gemini recibida para {estacion_nombre}.")

        return markdown.markdown(response.text)
    except Exception as e:
        print(f"Error al conectar con la API de Gemini para {estacion_nombre}: {e}")
        return f"Error al obtener recomendaciones de Gemini para {estacion_nombre}: {e}. Asegúrate de que tu clave API sea válida y tengas conexión a internet. Considera probar con 'gemini-1.0-pro' si 'gemini-2.5-flash' no está disponible."


# --- FUNCIÓN CENTRAL DE PREDICCIÓN Y ANÁLISIS ---
def predecir_y_analizar_precipitacion_sarima(ruta_csv, estacion_columna, anios_a_predecir=2, test_size_ratio=0.2, separator=','):
    metrics = {}
    future_predictions_for_html_df = pd.DataFrame()
    # monthly_avg_df = pd.Series() # Eliminado
    grafico_serie_temporal = ''
    # grafico_promedio_mensual = '' # Eliminado
    grafico_prediccion_sarima = ''

    # Asegurarse de que el directorio 'static' existe antes de guardar cualquier archivo
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Directorio 'static' creado.")


    try:
        df = pd.read_csv(ruta_csv, sep=separator)
        print(f"DataFrame cargado desde {ruta_csv}. Columnas: {df.columns.tolist()}")

        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.set_index('Fecha')
        df = df.sort_index()

        df[estacion_columna] = pd.to_numeric(df[estacion_columna], errors='coerce')

        # Interpolar y rellenar valores faltantes
        df[estacion_columna] = df[estacion_columna].interpolate(method='linear')
        df[estacion_columna] = df[estacion_columna].fillna(df[estacion_columna].mean())

        # Asegurarse de que el índice de tiempo esté completo y rellenar si hay huecos
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        df = df.reindex(full_date_range)
        df[estacion_columna] = df[estacion_columna].interpolate(method='linear')
        df[estacion_columna] = df[estacion_columna].fillna(df[estacion_columna].mean())

        y = df[estacion_columna]
        print(f"\nDatos de precipitación para {estacion_columna} cargados y preparados. Rango de fechas: {y.index.min()} a {y.index.max()}")
        print(f"Cantidad de puntos de datos: {len(y)}")

        print(f"\n--- INICIANDO ANÁLISIS DE DATOS Y PREPARACIÓN DE GRÁFICOS PARA {estacion_columna} ---")

        # # Calcular promedio mensual # Eliminado
        # monthly_avg_df = y.groupby(y.index.month).mean()
        # meses_nombres = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
        #                           7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        # monthly_avg_df = monthly_avg_df.rename(index=meses_nombres)

        # Gráfico de la serie temporal original
        plt.figure(figsize=(14, 7))
        plt.plot(y.index, y, label=f'Precipitación {estacion_columna}', color='blue')
        plt.title(f'Serie de Tiempo de Precipitación Mensual ({estacion_columna})')
        plt.xlabel('Fecha')
        plt.ylabel(f'Precipitación {estacion_columna} (mm)')
        plt.grid(True)
        plt.legend()
        plt.tick_params(axis='x', rotation=45)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.tight_layout()
        # Asegúrate de que los límites del eje y sean adecuados, evitando valores negativos si no tiene sentido
        plt.ylim(max(0, y.min() - 10), y.max() + 10) # Ajustado para no mostrar valores negativos si no tiene sentido
        
        # Corrección aquí: Usar os.path.join para la ruta completa
        grafico_serie_temporal_path = os.path.join('static', f'serie_temporal_precipitacion_{estacion_columna}.png')
        plt.savefig(grafico_serie_temporal_path)
        plt.close()
        grafico_serie_temporal = f'serie_temporal_precipitacion_{estacion_columna}.png' # Solo el nombre del archivo para el HTML
        print(f"Gráfico de la serie temporal original guardado como '{grafico_serie_temporal_path}'")


        # # Gráfico de barras de precipitación promedio mensual # Eliminado
        # plt.figure(figsize=(10, 6))
        # meses_para_barras = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        # # Asegúrate de que monthly_avg_df.values tiene el mismo orden que meses_para_barras
        # # Si monthly_avg_df no está ordenado por mes, esto podría causar un desajuste.
        # # Una forma más segura es usar reindex
        # monthly_avg_sorted = monthly_avg_df.reindex(list(meses_nombres.values()))
        # plt.bar(meses_para_barras, monthly_avg_sorted.values, color='skyblue') # Usar monthly_avg_sorted
        
        # plt.title(f'Precipitación Promedio Mensual ({estacion_columna} - Patrón Estacional)')
        # plt.xlabel('Mes')
        # plt.ylabel('Precipitación Promedio (mm)')
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # # Corrección aquí: Usar os.path.join para la ruta completa
        # grafico_promedio_mensual_path = os.path.join('static', f'precipitacion_mensual_promedio_barras_{estacion_columna}.png')
        # plt.savefig(grafico_promedio_mensual_path)
        # plt.close()
        # grafico_promedio_mensual = f'precipitacion_mensual_promedio_barras_{estacion_columna}.png' # Solo el nombre del archivo para el HTML
        # print(f"Gráfico de barras de precipitación promedio mensual guardado como '{grafico_promedio_mensual_path}'")


        print(f"\n--- INICIANDO MODELADO SARIMA Y PREDICCIONES PARA {estacion_columna} ---")

        # División de datos en entrenamiento y prueba
        test_size = int(len(y) * test_size_ratio)
        if test_size == 0 and len(y) > 0:
            test_size = 1
        elif len(y) == 0:
            raise ValueError("No hay datos para dividir en entrenamiento y prueba.")

        train = y[:-test_size]
        test = y[-test_size:]

        print(f"\nDividiendo datos para {estacion_columna}: {len(train)} para entrenamiento, {len(test)} para prueba.")

        # Búsqueda automática de parámetros SARIMA
        print(f"\nIniciando búsqueda automática de parámetros SARIMA para {estacion_columna} con auto_arima. Esto puede tomar tiempo...")

        stepwise_model = pm.auto_arima(train,
                                        start_p=0, start_q=0,
                                        test='adf',
                                        max_p=5, max_q=5,
                                        m=12,
                                        start_P=0, start_Q=0,
                                        max_P=2, max_Q=2,
                                        seasonal=True,
                                        D=1, # Orden de diferenciación estacional
                                        trace=False,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=True)

        best_order = stepwise_model.order
        best_seasonal_order = stepwise_model.seasonal_order
        print(f"Parámetros No Estacionales (p,d,q) para {estacion_columna}: {best_order}")
        print(f"Parámetros Estacionales (P,D,Q,s) para {estacion_columna}: {best_seasonal_order}")

        # Ajustar el modelo SARIMA
        model = sm.tsa.statespace.SARIMAX(train,
                                           order=best_order,
                                           seasonal_order=best_seasonal_order,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False)
        results = model.fit(disp=False)

        # Predicciones en el conjunto de prueba
        forecast_test = results.get_forecast(steps=len(test))
        forecast_test_mean = forecast_test.predicted_mean
        forecast_test_ci = forecast_test.conf_int(alpha=0.05)

        print(f"\nPredicciones generadas para el conjunto de prueba de {estacion_columna}.")

        print(f"\n--- Métricas de Evaluación en el Conjunto de Prueba para {estacion_columna} ---")
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

        # Predicciones futuras
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

        # Preparar DataFrame para CSV (incluye intervalos de confianza)
        df_para_csv = pd.DataFrame({
            'Fecha': forecast_future_mean.index.strftime('%Y-%m-%d'),
            f'{estacion_columna}_Predicho': forecast_future_mean.round(3).values
        })
        nombre_archivo_salida = f"Predicciones_Precipitacion_Futura_{estacion_columna}_{anios_a_predecir}anios.csv"
        df_para_csv.to_csv(nombre_archivo_salida, index=False)
        print(f"\nPredicciones futuras (completas) guardadas en '{nombre_archivo_salida}'")

        # Preparar DataFrame para HTML (solo fecha y predicción)
        future_predictions_for_html_df = pd.DataFrame({
            'Fecha': forecast_future_mean.index.strftime('%Y-%m-%d'),
            f'{estacion_columna}_Predicho': forecast_future_mean.round(3).values
        })

        # Gráfico final de datos y predicciones
        plt.figure(figsize=(16, 8))
        plt.plot(train.index, train, label=f'Datos de Entrenamiento ({estacion_columna})', color='blue')
        plt.plot(test.index, test, label=f'Datos Reales de Prueba ({estacion_columna})', color='green')
        plt.plot(forecast_test_mean.index, forecast_test_mean, label=f'Predicción en Prueba ({estacion_columna})', color='orange', linestyle='--')
        plt.fill_between(forecast_test_ci.index,
                         forecast_test_ci.iloc[:, 0],
                         forecast_test_ci.iloc[:, 1], color='orange', alpha=0.1, label='IC Predicción Prueba (95%)')

        plt.plot(forecast_future_mean.index, forecast_future_mean, label=f'Predicción Futura ({estacion_columna})', color='red')
        plt.fill_between(forecast_future_ci.index,
                         forecast_future_ci.iloc[:, 0],
                         forecast_future_ci.iloc[:, 1], color='pink', alpha=0.2, label='IC Predicción Futura (95%)')

        plt.title(f'Precipitación Mensual: Datos, Predicción en Prueba y Predicción Futura SARIMA ({estacion_columna} - {anios_a_predecir} años)')
        plt.xlabel('Fecha')
        plt.ylabel(f'Precipitación {estacion_columna} (mm)')
        plt.legend()
        plt.grid(True)
        plt.tick_params(axis='x', rotation=45)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.tight_layout()
        plt.ylim(max(0, y.min() - 10), y.max() + 10) # Ajustado para no mostrar valores negativos si no tiene sentido
        
        # Corrección aquí: Usar os.path.join para la ruta completa
        grafico_prediccion_sarima_path = os.path.join('static', f'prediccion_precipitacion_sarima_{estacion_columna}.png')
        plt.savefig(grafico_prediccion_sarima_path)
        plt.close()
        grafico_prediccion_sarima = f'prediccion_precipitacion_sarima_{estacion_columna}.png' # Solo el nombre del archivo para el HTML
        print(f"Gráfico de predicciones del modelo SARIMA guardado como '{grafico_prediccion_sarima_path}'")

        return {
            'metrics': metrics,
            'future_predictions_df': future_predictions_for_html_df,
            # 'monthly_avg_df': monthly_avg_df, # Eliminado
            'grafico_serie_temporal': grafico_serie_temporal, # Nombre del archivo para el HTML
            # 'grafico_promedio_mensual': grafico_promedio_mensual, # Eliminado
            'grafico_prediccion_sarima': grafico_prediccion_sarima # Nombre del archivo para el HTML
        }

    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_csv}' no se encontró.")
        return None
    except ValueError as ve:
        print(f"Error de datos para {estacion_columna}: {ve}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado para {estacion_columna}: {e}")
        import traceback
        traceback.print_exc() # Esto imprimirá el rastro completo del error
        return None

# --- Rutas de los archivos CSV y las columnas correspondientes ---
ESTACIONES_CONFIG = {
    'P55': {
        'ruta_csv': 'Precipitacion_Mensual__P55.csv',
        'columna_dato': 'P55',
        'separador': ','
    },
    'P43': {
        'ruta_csv': 'P43.csv',
        'columna_dato': 'P43',
        'separador': ';' # Observa que el segundo archivo usa ';'
    }
}

@app.route('/')
def index():
    resultados_estaciones = {}
    for estacion_nombre, config in ESTACIONES_CONFIG.items():
        print(f"\n--- PROCESANDO ESTACIÓN: {estacion_nombre} ---")
        result = predecir_y_analizar_precipitacion_sarima(
            config['ruta_csv'],
            config['columna_dato'],
            ANIOS_A_PREDECIR,
            TEST_SIZE_RATIO,
            config['separador']
        )
        if result:
            # Obtener recomendaciones de Gemini para cada estación
            gemini_recommendations = "Cargando recomendaciones..."
            if GOOGLE_API_KEY:
                gemini_recommendations = obtener_recomendaciones_gemini(
                    result['future_predictions_df'],
                    result['metrics'],
                    estacion_nombre
                )
            else:
                gemini_recommendations = "Por favor, configura tu GOOGLE_API_KEY en el archivo .env para obtener recomendaciones de IA."

            resultados_estaciones[estacion_nombre] = {
                'metrics': result['metrics'],
                'future_predictions_html': result['future_predictions_df'].to_html(classes='table table-striped table-bordered', index=False),
                # 'monthly_avg_html': result['monthly_avg_df'].to_frame(name=f'{estacion_nombre}_Promedio').to_html(classes='table table-striped table-bordered'), # Eliminado
                'grafico_serie_temporal': result['grafico_serie_temporal'],
                # 'grafico_promedio_mensual': result['grafico_promedio_mensual'], # Eliminado
                'grafico_prediccion_sarima': result['grafico_prediccion_sarima'],
                'gemini_recommendations': gemini_recommendations
            }
        else:
            resultados_estaciones[estacion_nombre] = {
                'error': f"No se pudieron procesar los datos para la estación {estacion_nombre}. Consulta la consola para más detalles."
            }

    return render_template('index.html', resultados=resultados_estaciones)


if __name__ == '__main__':
    app.run(debug=True)