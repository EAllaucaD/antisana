import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Usar el backend 'Agg' para guardar figuras sin mostrar la interfaz gráfica
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os
import warnings
import collections
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai # Importación de la librería

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("La variable de entorno 'GEMINI_API_KEY' no está configurada. Crea un archivo .env con tu clave.")
genai.configure(api_key=GEMINI_API_KEY)

# --- INICIO DE LA MODIFICACIÓN PARA LA SELECCIÓN DEL MODELO ---
# Inicializar el modelo de Gemini
selected_model_name = None
# Orden de preferencia para los modelos de Gemini
preferred_models = ["gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"] 

try:
    print("Listando modelos disponibles de Gemini...")
    available_models = list(genai.list_models())
    
    for preferred in preferred_models:
        for m in available_models:
            if preferred in m.name and "generateContent" in m.supported_generation_methods:
                selected_model_name = m.name
                break
        if selected_model_name:
            break
            
    if selected_model_name:
        gemini_model = genai.GenerativeModel(selected_model_name)
        print(f"Se utilizará el modelo de Gemini: {selected_model_name}")
    else:
        for m in available_models:
            if "generateContent" in m.supported_generation_methods and "text" in m.name:
                selected_model_name = m.name
                gemini_model = genai.GenerativeModel(selected_model_name)
                print(f"No se encontraron modelos preferidos. Usando: {selected_model_name}")
                break
        if not selected_model_name:
            raise ValueError("No se encontró ningún modelo de Gemini compatible para generar contenido de texto.")

except Exception as e:
    raise ValueError(f"Error al listar o inicializar modelos de Gemini: {e}. Asegúrate de que tu clave API sea válida y tengas conexión a internet.")
# --- FIN DE LA MODIFICACIÓN ---

# Configuración para evitar advertencias de matplotlib y pandas
warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Configura la ruta de tu archivo CSV y los parámetros de predicción ---
RUTA_CSV = 'P55_5.csv'
ANIOS_A_PREDECIR = 3 # Esto corresponde a los 36 meses
TEST_SIZE_RATIO = 0.2
NOMBRE_CSV_PREDICCIONES_FUTURAS = 'forest_predictions.csv' # Nombre para el CSV de predicciones futuras

# Variables globales para almacenar predicciones y métricas
global_future_predictions_df = pd.DataFrame()
global_metrics_ml = {}
global_plot_path_test = 'static/prediccion_precipitacion_ml_test.png'
global_plot_path_future = 'static/prediccion_precipitacion_ml_future.png'
global_plot_path_monthly_avg = 'static/precipitacion_mensual_promedio_barras.png' # Nueva ruta para el gráfico de promedio mensual

# --- 1. Carga y Preprocesamiento de Datos ---
def load_and_preprocess_data(ruta_csv):
    """
    Carga el archivo CSV, lo preprocesa (fechas, nulos, ordena) y retorna la serie de precipitación.
    """
    try:
        df = pd.read_csv(ruta_csv, sep=';', na_values=[''])
        print(f"DataFrame cargado. Columnas: {df.columns.tolist()}")

        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.set_index('Fecha')
        df = df.sort_index()

        df['P55_5'] = pd.to_numeric(df['P55_5'], errors='coerce') 

        # Asegura que no falte ningún mes en el índice y rellena NaNs
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        df = df.reindex(full_date_range)
        
        df['P55_5'] = df['P55_5'].interpolate(method='linear')
        df['P55_5'] = df['P55_5'].fillna(df['P55_5'].mean()) # Rellena cualquier NaN restante con la media

        print(f"\nDatos de precipitación cargados y preparados. Rango de fechas: {df.index.min()} a {df.index.max()}")
        print(f"Cantidad de puntos de datos: {len(df)}")
        return df['P55_5'] # Retornamos solo la serie de precipitación

    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_csv}' no se encontró.")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la carga de datos: {e}")
        return None

# --- 2. Ingeniería de Características para ML (Univariante) ---
def create_features_for_ml_univariate(y_series_full, lags, rolling_windows):
    """
    Crea características de series temporales para Machine Learning (univariante).
    """
    df_features = pd.DataFrame(index=y_series_full.index)
    df_features['target'] = y_series_full

    # 1. Características temporales
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

    # 2. Valores retrasados (Lags) de la columna objetivo
    for lag in lags:
        df_features[f'target_lag_{lag}'] = df_features['target'].shift(lag)

    # 3. Estadísticas de ventana móvil de la columna objetivo
    for window in rolling_windows:
        df_features[f'target_rolling_mean_{window}'] = df_features['target'].shift(1).rolling(window=window).mean()
        df_features[f'target_rolling_std_{window}'] = df_features['target'].shift(1).rolling(window=window).std()

    # Eliminar filas con NaN introducidos por los lags/rolling windows (las primeras filas)
    df_features = df_features.dropna()
    return df_features

# --- Definición de Lags y Ventanas para la precipitación ---
Lags_precipitacion = [1, 2, 3, 6, 12, 24]
Rolling_windows_precipitacion = [3, 6, 12]
MAX_LAG_NEEDED = max(Lags_precipitacion + Rolling_windows_precipitacion)

# --- Función para entrenar el modelo y generar predicciones ---
def train_and_predict():
    """
    Carga datos, entrena el modelo, evalúa y genera predicciones futuras.
    Actualiza variables globales y guarda gráficos.
    """
    global global_future_predictions_df, global_metrics_ml, global_plot_path_monthly_avg

    y_series = load_and_preprocess_data(RUTA_CSV)
    if y_series is None:
        return False # Indicar que hubo un error

    # --- Generar gráfico de promedio mensual ---
    monthly_avg_df = y_series.groupby(y_series.index.month).mean()
    meses_nombres = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                     7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    monthly_avg_df = monthly_avg_df.rename(index=meses_nombres)

    plt.figure(figsize=(10, 6))
    plt.bar(monthly_avg_df.index, monthly_avg_df.values, color='skyblue')
    plt.title('Precipitación Promedio Mensual (Patrón Estacional)')
    plt.xlabel('Mes')
    plt.ylabel('Precipitación Promedio P55_5 (mm)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(global_plot_path_monthly_avg) 
    plt.close()
    print(f"Gráfico de barras de precipitación promedio mensual guardado como '{global_plot_path_monthly_avg}'")


    df_ml = create_features_for_ml_univariate(y_series, Lags_precipitacion, Rolling_windows_precipitacion)
    
    X = df_ml.drop('target', axis=1)
    y = df_ml['target']

    split_index = int(len(X) * (1 - TEST_SIZE_RATIO))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    print("\nEntrenando el modelo RandomForestRegressor...")
    model.fit(X_train, y_train)
    print("Modelo entrenado.")

    # --- 5. Predicciones y Evaluación ---
    predictions_test = model.predict(X_test)

    global_metrics_ml = {
        'MSE': f"{mean_squared_error(y_test, predictions_test):.3f}",
        'RMSE': f"{np.sqrt(mean_squared_error(y_test, predictions_test)):.3f}",
        'MAE': f"{mean_absolute_error(y_test, predictions_test):.3f}",
        'R2': f"{r2_score(y_test, predictions_test):.3f}",
        'Correlación de Pearson': f"{pearsonr(y_test, predictions_test)[0]:.3f}"
    }
    print("\n--- Métricas de Evaluación (Machine Learning - Random Forest) ---")
    print(global_metrics_ml)

    # --- Visualización de Predicciones en el Conjunto de Prueba ---
    plt.figure(figsize=(16, 8))
    plt.plot(y_train.index, y_train, label='Entrenamiento Real', color='blue')
    plt.plot(y_test.index, y_test, label='Prueba Real', color='green')
    plt.plot(y_test.index, predictions_test, label='Predicciones ML (Random Forest)', color='red', linestyle='--')
    plt.title('Precipitación Mensual P55_5: Datos Reales y Predicciones con Random Forest')
    plt.xlabel('Fecha')
    plt.ylabel('Precipitación P55_5 (mm)')
    plt.legend()
    plt.grid(True)
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    # Ajusta el límite Y para que se vea bien, pero no corte los datos
    plt.ylim(min(y_series.min(), predictions_test.min()) - 10, max(y_series.max(), predictions_test.max()) + 10) 
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig(global_plot_path_test)
    plt.close()
    print("Gráfico de predicciones del modelo ML en conjunto de prueba guardado.")

    # --- 6. Predicción a Futuro (Recursiva) ---
    print(f"\n--- Iniciando Predicción Futura para {ANIOS_A_PREDECIR * 12} meses (Random Forest) ---")

    num_future_months = ANIOS_A_PREDECIR * 12
    future_dates_ml = pd.date_range(start=y_series.index.max() + pd.DateOffset(months=1), periods=num_future_months, freq='MS')
    
    # Asegúrate de que las fechas futuras se extiendan hasta la fecha esperada.
    # Si y_series.index.max() es 2023-06-30, entonces la primera fecha predicha será 2023-07-01
    # Y los 36 periodos llevarán hasta 2026-06-01 (junio 2026).
    print(f"La serie de predicciones futuras comenzará en: {future_dates_ml.min().strftime('%Y-%m-%d')}")
    print(f"Y terminará en: {future_dates_ml.max().strftime('%Y-%m-%d')}")

    future_predictions_ml = []
    
    # Necesitamos un "buffer" o historial de los últimos valores para calcular los lags y rolling stats.
    history_buffer = collections.deque(y_series.tail(MAX_LAG_NEEDED).tolist(), maxlen=MAX_LAG_NEEDED)

    for i in range(num_future_months):
        current_future_date = future_dates_ml[i]
        
        # 1. Crear las características para el mes actual de la predicción
        current_features_dict = {}
        
        # Características temporales
        current_features_dict['month'] = current_future_date.month
        current_features_dict['year'] = current_future_date.year
        current_features_dict['quarter'] = current_future_date.quarter
        current_features_dict['month_sin'] = np.sin(2 * np.pi * current_future_date.month / 12)
        current_features_dict['month_cos'] = np.cos(2 * np.pi * current_future_date.month / 12)

        # Lags de precipitación (usando el history_buffer)
        for lag_val in Lags_precipitacion:
            if len(history_buffer) >= lag_val:
                current_features_dict[f'target_lag_{lag_val}'] = history_buffer[-lag_val]
            else:
                # Si no hay suficientes datos en el buffer para un lag, usar la media histórica como fallback
                current_features_dict[f'target_lag_{lag_val}'] = y_series.mean()

        # Estadísticas de ventana móvil (usando el history_buffer)
        temp_series = pd.Series(list(history_buffer)) 
        for window in Rolling_windows_precipitacion:
            if len(temp_series) >= window:
                current_features_dict[f'target_rolling_mean_{window}'] = temp_series.tail(window).mean()
                current_features_dict[f'target_rolling_std_{window}'] = temp_series.tail(window).std()
            else:
                # Si no hay suficientes datos en el buffer para la ventana, usar la media/std histórica como fallback
                current_features_dict[f'target_rolling_mean_{window}'] = y_series.mean()
                current_features_dict[f'target_rolling_std_{window}'] = y_series.std()

        # Convertir el diccionario de características a un DataFrame con el orden correcto de columnas
        future_features_df_single_row = pd.DataFrame([current_features_dict])
        future_features_df_single_row = future_features_df_single_row.reindex(columns=X_train.columns, fill_value=0)

        # Realizar la predicción
        next_predicted_value = model.predict(future_features_df_single_row)[0]
        
        # Asegurar que las predicciones no sean negativas
        next_predicted_value = max(0, next_predicted_value) 
        
        future_predictions_ml.append(next_predicted_value)

        # Actualizar el history_buffer con la nueva predicción para el siguiente paso
        history_buffer.append(next_predicted_value)

    # Crear DataFrame final de predicciones futuras
    global_future_predictions_df = pd.DataFrame({
        'Fecha': future_dates_ml,
        'P55_5_Predicho': [f"{val:.2f}" for val in future_predictions_ml] # Formatear a 2 decimales
    })

    # AL GUARDAR EN CSV, TAMBIÉN NOS ASEGURAMOS DEL FORMATO
    global_future_predictions_df.to_csv(NOMBRE_CSV_PREDICCIONES_FUTURAS, index=False, date_format='%Y-%m-%d')
    print(f"\nPredicciones futuras del modelo ML (Random Forest) guardadas en '{NOMBRE_CSV_PREDICCIONES_FUTURAS}'")

    # --- Visualización de Predicciones Futuras ---
    plt.figure(figsize=(16, 8))
    plt.plot(y_series.index, y_series, label='Datos Históricos (P55_5)', color='blue')
    # Convertir las predicciones a float para graficar si se almacenaron como string
    plt.plot(global_future_predictions_df['Fecha'], global_future_predictions_df['P55_5_Predicho'].astype(float), 
             label=f'Predicción Futura ML (Random Forest - {ANIOS_A_PREDECIR} años)', color='red', linestyle='--')
    plt.title(f'Precipitación Mensual P55_5: Datos Históricos y Predicción Futura con Random Forest')
    plt.xlabel('Fecha')
    plt.ylabel('Precipitación P55_5 (mm)')
    plt.legend()
    plt.grid(True)
    plt.tick_params(axis='x', rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.savefig(global_plot_path_future)
    plt.close()
    print("Gráfico de predicciones futuras del modelo ML guardado como 'static/prediccion_precipitacion_ml_future.png'")

    return True

# --- Inicialización del Modelo ---
print("Inicializando el modelo y generando predicciones...")
if not train_and_predict():
    print("Error al inicializar el modelo, saliendo.")
    exit()
print("Modelo inicializado y predicciones generadas con éxito.")


# --- Rutas de Flask ---
@app.route('/')
def index():
    predictions_display_df = global_future_predictions_df.copy()
    predictions_display_df['Fecha'] = predictions_display_df['Fecha'].dt.strftime('%Y-%m-%d')
    predictions_html = predictions_display_df.to_html(index=False, classes='table table-striped table-hover')
    
    return render_template('index.html', 
                           metrics=global_metrics_ml, 
                           predictions_html=predictions_html,
                           plot_test=global_plot_path_test,
                           plot_future=global_plot_path_future,
                           plot_monthly_avg=global_plot_path_monthly_avg, # Pasando la nueva ruta del gráfico
                           plot_location='static/Imagen1.png', # Ruta a la imagen de ubicación
                           ANIOS_A_PREDECIR=ANIOS_A_PREDECIR)

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    """
    Endpoint para obtener recomendaciones de la IA de Gemini basadas en las predicciones.
    """
    # Determinar el rango de fechas de las predicciones
    start_date_pred = global_future_predictions_df['Fecha'].min().strftime('%Y-%m-%d')
    end_date_pred = global_future_predictions_df['Fecha'].max().strftime('%Y-%m-%d')

    # Formular el prompt para Gemini
    prompt_base = f"""
    Eres un asistente experto en agricultura y gestión de riesgos, especializado en precipitación.
    Te proporcionaré datos de precipitación mensual predicha (en milímetros) para los próximos {ANIOS_A_PREDECIR * 12} meses, específicamente desde el {start_date_pred} hasta el {end_date_pred}.
    Genera recomendaciones concisas y útiles para:
    1.  **Agricultores:** Qué cultivos considerar, manejo de riego, siembras, cosechas, prevención de enfermedades.
    2.  **Gestores de Plan de Riesgos:** Preparación ante sequías o inundaciones, gestión de recursos hídricos, alerta temprana, planificación de infraestructura.

    Los datos de precipitación mensual predicha son los siguientes:
    """
    
    # Usar todas las predicciones futuras (36 meses)
    relevant_predictions = global_future_predictions_df
    
    # Formatear las predicciones para el prompt, asegurando los 2 decimales
    formatted_predictions = "\n".join([f"- {row['Fecha']}: {row['P55_5_Predicho']} mm" # Ya está formateado a string con 2 decimales
                                         for index, row in relevant_predictions.iterrows()])
    
    full_prompt = f"{prompt_base}\n{formatted_predictions}\n\n" \
                  f"Por favor, estructura tus recomendaciones claramente en dos secciones: 'Para Agricultores' y 'Para Gestión de Riesgos'. Incluye sugerencias específicas para meses o periodos si los datos lo justifican."
    
    try:
        response = gemini_model.generate_content(full_prompt)
        ai_response = response.text
    except Exception as e:
        ai_response = f"Lo siento, hubo un error al comunicarse con la IA: {e}. Asegúrate de que tu clave API de Gemini sea válida y tengas conexión a internet."
        print(f"Error en Gemini API: {e}")

    return jsonify({'response': ai_response})

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)