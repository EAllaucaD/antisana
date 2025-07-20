import pandas as pd
import numpy as np
import matplotlib # ¡IMPORTA ESTO PRIMERO!
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os
import warnings
import collections # Para la cola de historial en la predicción futura

# Configuración para evitar advertencias de matplotlib y pandas
warnings.filterwarnings("ignore")
matplotlib.use('Agg') # Usar el backend 'Agg' para guardar figuras sin mostrar la interfaz gráfica

# --- Configura la ruta de tu archivo CSV y los parámetros de predicción ---
# Asegúrate de que tu CSV tenga las columnas 'Fecha' y 'P55_5'
RUTA_CSV = 'P55_5.csv' 
ANIOS_A_PREDECIR = 3 # Esto corresponde a los 24 meses
TEST_SIZE_RATIO = 0.2

# --- 1. Carga y Preprocesamiento de Datos ---
def load_and_preprocess_data(ruta_csv):
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

y_series = load_and_preprocess_data(RUTA_CSV)

if y_series is None:
    exit() # Salir si hubo un problema con la carga de datos

# --- 2. Ingeniería de Características para ML (Univariante) ---
def create_features_for_ml_univariate(y_series_full, lags, rolling_windows):
    """
    Crea características de series temporales para Machine Learning (univariante).

    Args:
        y_series_full (pd.Series): La serie temporal completa (ej. df['P55_5']), con índice de fecha.
        lags (list): Lista de lags a generar para la columna objetivo (ej. [1, 12, 24]).
        rolling_windows (list): Lista de tamaños de ventana para estadísticas móviles de la columna objetivo (ej. [3, 6, 12]).

    Returns:
        pd.DataFrame: DataFrame con características y la columna objetivo.
    """
    df_features = pd.DataFrame(index=y_series_full.index)
    df_features['target'] = y_series_full # Esta será nuestra 'y'

    # 1. Características temporales
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

    # 2. Valores retrasados (Lags) de la columna objetivo
    for lag in lags:
        # Usamos .shift(lag) para obtener el valor del mes anterior, etc.
        df_features[f'target_lag_{lag}'] = df_features['target'].shift(lag)

    # 3. Estadísticas de ventana móvil de la columna objetivo
    for window in rolling_windows:
        # shift(1) para asegurar que las estadísticas no usen el valor actual que queremos predecir
        df_features[f'target_rolling_mean_{window}'] = df_features['target'].shift(1).rolling(window=window).mean()
        df_features[f'target_rolling_std_{window}'] = df_features['target'].shift(1).rolling(window=window).std()
        # Puedes añadir min, max, median, etc. si lo consideras útil

    # Eliminar filas con NaN introducidos por los lags/rolling windows (las primeras filas)
    df_features = df_features.dropna()
    return df_features

# --- Definición de Lags y Ventanas para la precipitación ---
Lags_precipitacion = [1, 2, 3, 6, 12, 24] # Lags comunes: mes anterior, 2 meses, medio año, año anterior, 2 años anteriores
Rolling_windows_precipitacion = [3, 6, 12] # Ventanas para promedios y desviaciones estándar

# Encontrar el lag más grande y la ventana de rolling más grande para el historial
MAX_LAG_NEEDED = max(Lags_precipitacion + Rolling_windows_precipitacion) 

df_ml = create_features_for_ml_univariate(y_series, Lags_precipitacion, Rolling_windows_precipitacion)

print(f"\nDataFrame con características creado. Primeras 5 filas:\n{df_ml.head()}")
print(f"Número de características generadas: {len(df_ml.columns) - 1}") # -1 por la columna 'target'
print(f"Cantidad de puntos de datos para el ML: {len(df_ml)}")

# --- 3. División de Datos (Cronológica) ---
X = df_ml.drop('target', axis=1) # Todas las características
y = df_ml['target']             # La precipitación que queremos predecir

split_index = int(len(X) * (1 - TEST_SIZE_RATIO))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"\nDatos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

# --- 4. Selección y Entrenamiento del Modelo ML (Random Forest Regressor) ---
# Puedes ajustar los hiperparámetros de RandomForestRegressor para optimizar el rendimiento:
# n_estimators: número de árboles en el bosque (más árboles suelen ser mejores pero más lentos)
# max_depth: profundidad máxima de cada árbol (controla el overfitting)
# min_samples_leaf: número mínimo de muestras requeridas para estar en un nodo hoja (controla el overfitting)
# random_state: para reproducibilidad de resultados
# n_jobs=-1: usa todos los núcleos de la CPU disponibles
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)

print("\nEntrenando el modelo RandomForestRegressor...")
model.fit(X_train, y_train)
print("Modelo entrenado.")

# --- 5. Predicciones y Evaluación ---
predictions_test = model.predict(X_test)

metrics_ml = {
    'MSE': f"{mean_squared_error(y_test, predictions_test):.3f}",
    'RMSE': f"{np.sqrt(mean_squared_error(y_test, predictions_test)):.3f}",
    'MAE': f"{mean_absolute_error(y_test, predictions_test):.3f}",
    'R2': f"{r2_score(y_test, predictions_test):.3f}",
    'Correlación de Pearson': f"{pearsonr(y_test, predictions_test)[0]:.3f}"
}

print("\n--- Métricas de Evaluación (Machine Learning - Random Forest) ---")
print(metrics_ml)

# --- Visualización de Predicciones en el Conjunto de Prueba ---
plt.figure(figsize=(16, 8))
plt.plot(y_train.index, y_train, label='Entrenamiento Real', color='blue')
plt.plot(y_test.index, y_test, label='Prueba Real', color='green')
plt.plot(y_test.index, predictions_test, label='Predicciones ML (Random Forest)', color='red', linestyle='--')
plt.title('Precipitación Mensual: Datos Reales y Predicciones con Random Forest')
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
plt.savefig('static/prediccion_precipitacion_ml_test.png')
plt.close()
print("Gráfico de predicciones del modelo ML en conjunto de prueba guardado como 'static/prediccion_precipitacion_ml_test.png'")


# --- 6. Predicción a Futuro (Recursiva) ---
print(f"\n--- Iniciando Predicción Futura para {ANIOS_A_PREDECIR * 12} meses (Random Forest) ---")

num_future_months = ANIOS_A_PREDECIR * 12
future_predictions_ml = []
future_dates_ml = pd.date_range(start=y_series.index.max() + pd.DateOffset(months=1), periods=num_future_months, freq='MS')

# Necesitamos un "buffer" o historial de los últimos valores para calcular los lags y rolling stats.
# Se usará una deque para manejar eficientemente los últimos N valores.
# El tamaño del deque debe ser al menos el máximo lag o el máximo tamaño de ventana.
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
            # Esto no debería ocurrir si history_buffer tiene el tamaño correcto,
            # pero es un fallback para seguridad.
            current_features_dict[f'target_lag_{lag_val}'] = y_series.mean() # O algún otro valor de relleno

    # Estadísticas de ventana móvil (usando el history_buffer)
    # Se crea una serie temporal temporal a partir del buffer para calcular las rolling stats.
    temp_series = pd.Series(list(history_buffer)) # Convierte deque a lista y luego a Series
    for window in Rolling_windows_precipitacion:
        # Aseguramos que haya suficientes datos para el cálculo de la ventana
        if len(temp_series) >= window:
            current_features_dict[f'target_rolling_mean_{window}'] = temp_series.tail(window).mean()
            current_features_dict[f'target_rolling_std_{window}'] = temp_series.tail(window).std()
        else:
            # Si no hay suficientes datos en el buffer para la ventana, usa valores por defecto
            current_features_dict[f'target_rolling_mean_{window}'] = y_series.mean()
            current_features_dict[f'target_rolling_std_{window}'] = y_series.std()

    # Convertir el diccionario de características a un DataFrame con el orden correcto de columnas
    future_features_df_single_row = pd.DataFrame([current_features_dict])
    # Asegurarse de que las columnas estén en el mismo orden que las usadas para entrenar (X_train.columns)
    future_features_df_single_row = future_features_df_single_row.reindex(columns=X_train.columns, fill_value=0) # Rellenar con 0 si falta alguna

    # Realizar la predicción
    next_predicted_value = model.predict(future_features_df_single_row)[0]
    
    # Asegurar que las predicciones no sean negativas (la precipitación no puede ser negativa)
    next_predicted_value = max(0, next_predicted_value) 
    
    future_predictions_ml.append(next_predicted_value)

    # Actualizar el history_buffer con la nueva predicción para el siguiente paso
    history_buffer.append(next_predicted_value)

# Crear DataFrame final de predicciones futuras
final_future_predictions_df = pd.DataFrame({
    'Fecha': future_dates_ml.strftime('%Y-%m-%d'),
    'P55_5_Predicho': future_predictions_ml
})

nombre_archivo_salida_ml_futuro = f"Predicciones_Precipitacion_Futura_ML_P55_5_{ANIOS_A_PREDECIR}anios.csv"
final_future_predictions_df.to_csv(nombre_archivo_salida_ml_futuro, index=False)
print(f"\nPredicciones futuras del modelo ML (Random Forest) guardadas en '{nombre_archivo_salida_ml_futuro}'")

# --- Visualización de Predicciones Futuras ---
plt.figure(figsize=(16, 8))
plt.plot(y_series.index, y_series, label='Datos Históricos (P55_5)', color='blue')
plt.plot(final_future_predictions_df['Fecha'], final_future_predictions_df['P55_5_Predicho'], 
         label=f'Predicción Futura ML (Random Forest - {ANIOS_A_PREDECIR} años)', color='red', linestyle='--')
plt.title(f'Precipitación Mensual: Datos Históricos y Predicción Futura con Random Forest')
plt.xlabel('Fecha')
plt.ylabel('Precipitación P55_5 (mm)')
plt.legend()
plt.grid(True)
plt.tick_params(axis='x', rotation=45)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig('static/prediccion_precipitacion_ml_future.png')
plt.close()
print("Gráfico de predicciones futuras del modelo ML guardado como 'static/prediccion_precipitacion_ml_future.png'")