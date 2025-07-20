import pandas as pd
import numpy as np
import matplotlib # Importa matplotlib primero
matplotlib.use('Agg') # Usar el backend 'Agg' para guardar figuras sin mostrar la interfaz gráfica
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit # Importaciones nuevas para optimización
import os
import warnings
import collections # Para la cola de historial en la predicción futura

# Configuración para evitar advertencias de matplotlib y pandas
warnings.filterwarnings("ignore")

# --- Configura la ruta de tu archivo CSV y los parámetros de predicción ---
# Asegúrate de que tu CSV tenga las columnas 'Fecha' y 'P55_5'
RUTA_CSV = 'P55_5.csv' 
ANIOS_A_PREDECIR = 2 # Esto corresponde a los 24 meses
TEST_SIZE_RATIO = 0.2

# --- 1. Carga y Preprocesamiento de Datos ---
def load_and_preprocess_data(ruta_csv):
    """
    Carga los datos desde un archivo CSV y realiza el preprocesamiento inicial.
    Asegura que el índice sea de tipo fecha y maneja valores faltantes.
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
        df_features[f'target_lag_{lag}'] = df_features['target'].shift(lag)

    # 3. Estadísticas de ventana móvil de la columna objetivo
    for window in rolling_windows:
        df_features[f'target_rolling_mean_{window}'] = df_features['target'].shift(1).rolling(window=window).mean()
        df_features[f'target_rolling_std_{window}'] = df_features['target'].shift(1).rolling(window=window).std()

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

# --- 4. Optimización de Hiperparámetros con GridSearchCV ---
print("\n--- Iniciando optimización de hiperparámetros con GridSearchCV ---")

# Define la rejilla de parámetros a probar
# Puedes ajustar estos rangos. Más valores = más tiempo de cómputo.
param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árboles
    'max_depth': [5, 10, 15],         # Profundidad máxima del árbol
    'min_samples_leaf': [1, 3, 5],    # Mínimo de muestras por hoja
    'max_features': [0.7, 0.9, 'sqrt'] # Proporción de características a considerar por árbol
}

# Inicializa el modelo base Random Forest
rf_model_base = RandomForestRegressor(random_state=42, n_jobs=-1)

# Configura TimeSeriesSplit para la validación cruzada adecuada para series temporales
# n_splits: número de divisiones de entrenamiento/validación.
# gap: número de muestras a excluir entre el conjunto de entrenamiento y el de validación (opcional, pero útil)
tscv = TimeSeriesSplit(n_splits=5) 

# Configura GridSearchCV
# scoring='r2' para optimizar el R2. Puedes cambiarlo a 'neg_mean_squared_error' para optimizar MSE (negativo porque GridSearchCV maximiza).
grid_search = GridSearchCV(estimator=rf_model_base, 
                           param_grid=param_grid, 
                           cv=tscv, 
                           scoring='r2', 
                           verbose=1, # Muestra el progreso de la búsqueda
                           n_jobs=-1) # Usa todos los núcleos de la CPU

# Entrena el GridSearchCV con tus datos de entrenamiento
grid_search.fit(X_train, y_train)

print("\n--- Búsqueda de hiperparámetros completada ---")
print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
print(f"Mejor puntuación (R2) en validación cruzada: {grid_search.best_score_:.3f}")

# Usa el mejor modelo encontrado por GridSearchCV para el resto del script
model = grid_search.best_estimator_
print("El modelo ahora es el mejor estimador encontrado por GridSearchCV.")

# --- 5. Predicciones y Evaluación ---
predictions_test = model.predict(X_test)

metrics_ml = {
    'MSE': f"{mean_squared_error(y_test, predictions_test):.3f}",
    'RMSE': f"{np.sqrt(mean_squared_error(y_test, predictions_test)):.3f}",
    'MAE': f"{mean_absolute_error(y_test, predictions_test):.3f}",
    'R2': f"{r2_score(y_test, predictions_test):.3f}",
    'Correlación de Pearson': f"{pearsonr(y_test, predictions_test)[0]:.3f}"
}

print("\n--- Métricas de Evaluación Finales (Machine Learning - Random Forest Optimizado) ---")
print(metrics_ml)

# --- Visualización de Predicciones en el Conjunto de Prueba ---
plt.figure(figsize=(16, 8))
plt.plot(y_train.index, y_train, label='Entrenamiento Real', color='blue')
plt.plot(y_test.index, y_test, label='Prueba Real', color='green')
plt.plot(y_test.index, predictions_test, label='Predicciones ML (Random Forest Optimizado)', color='red', linestyle='--')
plt.title('Precipitación Mensual: Datos Reales y Predicciones con Random Forest Optimizado')
plt.xlabel('Fecha')
plt.ylabel('Precipitación P55_5 (mm)')
plt.legend()
plt.grid(True)
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.ylim(min(y_series.min(), predictions_test.min()) - 10, max(y_series.max(), predictions_test.max()) + 10) 
if not os.path.exists('static'):
    os.makedirs('static')
plt.savefig('static/prediccion_precipitacion_ml_test_optimized.png')
plt.close()
print("Gráfico de predicciones del modelo ML optimizado en conjunto de prueba guardado como 'static/prediccion_precipitacion_ml_test_optimized.png'")


# --- 6. Predicción a Futuro (Recursiva) ---
print(f"\n--- Iniciando Predicción Futura para {ANIOS_A_PREDECIR * 12} meses (Random Forest Optimizado) ---")

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
            current_features_dict[f'target_lag_{lag_val}'] = y_series.mean() # Fallback si el buffer es muy pequeño

    # Estadísticas de ventana móvil (usando el history_buffer)
    temp_series = pd.Series(list(history_buffer)) # Convierte deque a lista y luego a Series
    for window in Rolling_windows_precipitacion:
        if len(temp_series) >= window:
            current_features_dict[f'target_rolling_mean_{window}'] = temp_series.tail(window).mean()
            current_features_dict[f'target_rolling_std_{window}'] = temp_series.tail(window).std()
        else:
            current_features_dict[f'target_rolling_mean_{window}'] = y_series.mean()
            current_features_dict[f'target_rolling_std_{window}'] = y_series.std()

    # Convertir el diccionario de características a un DataFrame con el orden correcto de columnas
    future_features_df_single_row = pd.DataFrame([current_features_dict])
    # Asegurarse de que las columnas estén en el mismo orden que las usadas para entrenar (X_train.columns)
    future_features_df_single_row = future_features_df_single_row.reindex(columns=X_train.columns, fill_value=0)

    # Realizar la predicción
    next_predicted_value = model.predict(future_features_df_single_row)[0]
    
    # Asegurar que las predicciones no sean negativas
    next_predicted_value = max(0, next_predicted_value) 
    
    future_predictions_ml.append(next_predicted_value)

    # Actualizar el history_buffer con la nueva predicción para el siguiente paso
    history_buffer.append(next_predicted_value)

# Crear DataFrame final de predicciones futuras
final_future_predictions_df = pd.DataFrame({
    'Fecha': future_dates_ml.strftime('%Y-%m-%d'),
    'P55_5_Predicho': future_predictions_ml
})

# --- Guardar en un CSV diferente para las predicciones optimizadas ---
nombre_archivo_salida_ml_futuro_optimizado = f"PreciPredictRF_Predicciones_Futuras_Optimizadas_{ANIOS_A_PREDECIR}anios.csv"
final_future_predictions_df.to_csv(nombre_archivo_salida_ml_futuro_optimizado, index=False)
print(f"\nPredicciones futuras del modelo ML (Random Forest Optimizado) guardadas en '{nombre_archivo_salida_ml_futuro_optimizado}'")

# --- Visualización de Predicciones Futuras ---
plt.figure(figsize=(16, 8))
plt.plot(y_series.index, y_series, label='Datos Históricos (P55_5)', color='blue')
plt.plot(final_future_predictions_df['Fecha'], final_future_predictions_df['P55_5_Predicho'], 
         label=f'Predicción Futura ML (PreciPredict RF - {ANIOS_A_PREDECIR} años)', color='red', linestyle='--')
plt.title(f'Precipitación Mensual: Datos Históricos y Predicción Futura con PreciPredict RF')
plt.xlabel('Fecha')
plt.ylabel('Precipitación P55_5 (mm)')
plt.legend()
plt.grid(True)
plt.tick_params(axis='x', rotation=45)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig('static/prediccion_precipitacion_ml_future_optimized.png')
plt.close()
print("Gráfico de predicciones futuras del modelo ML optimizado guardado como 'static/prediccion_precipitacion_ml_future_optimized.png'")