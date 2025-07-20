import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Usar el backend 'Agg' para guardar figuras sin mostrar la interfaz gráfica
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit # Importaciones clave para optimización
import os
import warnings
import collections # Para la cola de historial en la predicción futura

# Configuración para evitar advertencias de matplotlib y pandas
warnings.filterwarnings("ignore")

# --- Configura la ruta de tu archivo CSV y los parámetros de predicción ---
# ASEGÚRATE DE QUE TU CSV TENGA LAS COLUMNAS 'Fecha' y 'P43_33'
RUTA_CSV = 'P43_33.csv' # Archivo de datos de entrada
ANIOS_A_PREDECIR = 2 # Cantidad de años a predecir a futuro
TEST_SIZE_RATIO = 0.2 # Proporción de datos para el conjunto de prueba (ej. 0.2 = 20%)

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

        # Asegúrate de que la columna con los datos de precipitación sea 'P43_33'
        df['P43_33'] = pd.to_numeric(df['P43_33'], errors='coerce') 

        # Asegura que no falte ningún mes en el índice y rellena NaNs
        # Esto es crucial para series temporales con frecuencia mensual
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        df = df.reindex(full_date_range)
        
        df['P43_33'] = df['P43_33'].interpolate(method='linear') # Interpola valores faltantes
        df['P43_33'] = df['P43_33'].fillna(df['P43_33'].mean()) # Rellena cualquier NaN restante con la media (para los extremos)

        print(f"\nDatos de precipitación cargados y preparados. Rango de fechas: {df.index.min()} a {df.index.max()}")
        print(f"Cantidad de puntos de datos: {len(df)}")
        return df['P43_33'] # Retornamos solo la serie de precipitación

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
        y_series_full (pd.Series): La serie temporal completa (ej. df['P43_33']), con índice de fecha.
        lags (list): Lista de lags a generar para la columna objetivo (ej. [1, 12, 24]).
        rolling_windows (list): Lista de tamaños de ventana para estadísticas móviles de la columna objetivo (ej. [3, 6, 12]).

    Returns:
        pd.DataFrame: DataFrame con características y la columna objetivo.
    """
    df_features = pd.DataFrame(index=y_series_full.index)
    df_features['target'] = y_series_full # Esta será nuestra 'y' (la precipitación P43_33)

    # 1. Características temporales (estacionalidad, tendencia)
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter
    # Uso de seno/coseno para capturar la estacionalidad cíclica
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Podemos añadir una característica lineal para la tendencia a largo plazo
    # df_features['time_idx'] = np.arange(len(df_features)) 

    # 2. Valores retrasados (Lags) de la columna objetivo
    for lag in lags:
        # Usamos .shift(lag) para obtener el valor del mes anterior, año anterior, etc.
        df_features[f'target_lag_{lag}'] = df_features['target'].shift(lag)

    # 3. Estadísticas de ventana móvil de la columna objetivo
    for window in rolling_windows:
        # shift(1) para asegurar que las estadísticas no usen el valor actual que queremos predecir (evitar fuga de datos)
        df_features[f'target_rolling_mean_{window}'] = df_features['target'].shift(1).rolling(window=window).mean()
        df_features[f'target_rolling_std_{window}'] = df_features['target'].shift(1).rolling(window=window).std()
        # Puedes añadir otras estadísticas si lo consideras útil (ej. min, max, median, skew, kurt)
        # df_features[f'target_rolling_min_{window}'] = df_features['target'].shift(1).rolling(window=window).min()
        # df_features[f'target_rolling_max_{window}'] = df_features['target'].shift(1).rolling(window=window).max()
        # df_features[f'target_rolling_median_{window}'] = df_features['target'].shift(1).rolling(window=window).median()

    # Eliminar filas con NaN introducidos por los lags/rolling windows (las primeras filas del DataFrame)
    df_features = df_features.dropna()
    return df_features

# --- Definición de Lags y Ventanas para la precipitación ---
# Estos son los lags y ventanas que el modelo usará para crear sus características.
Lags_precipitacion = [1, 2, 3, 6, 12, 18, 24, 36] # He añadido lags 18 y 36 para mayor profundidad
Rolling_windows_precipitacion = [3, 6, 9, 12, 18] # He añadido ventanas 9 y 18 para mayor detalle

# Encontrar el lag más grande y la ventana de rolling más grande para el historial
# Esto asegura que el buffer de predicción futura tenga suficientes datos
MAX_LAG_NEEDED = max(Lags_precipitacion + Rolling_windows_precipitacion) 

df_ml = create_features_for_ml_univariate(y_series, Lags_precipitacion, Rolling_windows_precipitacion)

print(f"\nDataFrame con características creado. Primeras 5 filas:\n{df_ml.head()}")
print(f"Número de características generadas: {len(df_ml.columns) - 1}") # -1 por la columna 'target'
print(f"Cantidad de puntos de datos para el ML: {len(df_ml)}")

# --- 3. División de Datos (Cronológica) ---
X = df_ml.drop('target', axis=1) # Todas las características (variables explicativas)
y = df_ml['target']             # La precipitación que queremos predecir (variable objetivo)

# División cronológica: los datos más antiguos para entrenamiento, los más recientes para prueba.
split_index = int(len(X) * (1 - TEST_SIZE_RATIO))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"\nDatos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

# --- 4. Optimización de Hiperparámetros con GridSearchCV ---
print("\n--- Iniciando optimización de hiperparámetros con GridSearchCV ---")

# Define la rejilla de parámetros a probar para RandomForestRegressor
# He ampliado los rangos para una búsqueda más exhaustiva. Esto tomará más tiempo.
param_grid = {
    'n_estimators': [100, 200, 300, 400],  # Número de árboles en el bosque
    'max_depth': [5, 10, 15, 20, None],  # Profundidad máxima del árbol (None = sin límite)
    'min_samples_leaf': [1, 2, 3, 5, 10],    # Mínimo de muestras por hoja
    'max_features': [0.7, 0.9, 'sqrt', 'log2'], # Proporción de características a considerar por árbol
    # Puedes añadir otros como 'min_samples_split', 'bootstrap', etc.
}

# Inicializa el modelo base Random Forest
rf_model_base = RandomForestRegressor(random_state=42, n_jobs=-1)

# Configura TimeSeriesSplit para la validación cruzada adecuada para series temporales
# n_splits: número de divisiones de entrenamiento/validación. Más splits = más robusto, más lento.
# gap: número de muestras a excluir entre el conjunto de entrenamiento y el de validación (opcional, útil para evitar fugas)
tscv = TimeSeriesSplit(n_splits=5) 

# Configura GridSearchCV
# scoring='r2' para optimizar el R2. Puedes cambiarlo a 'neg_mean_squared_error' para optimizar MSE (negativo porque GridSearchCV maximiza).
grid_search = GridSearchCV(estimator=rf_model_base, 
                           param_grid=param_grid, 
                           cv=tscv, 
                           scoring='r2', 
                           verbose=2, # Muestra más detalles del progreso de la búsqueda
                           n_jobs=-1) # Usa todos los núcleos de la CPU disponibles

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
plt.title('Precipitación Mensual (P43_33): Datos Reales y Predicciones con Random Forest Optimizado')
plt.xlabel('Fecha')
plt.ylabel('Precipitación P43_33 (mm)')
plt.legend()
plt.grid(True)
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
# Ajusta el límite Y para que se vea bien, pero no corte los datos
plt.ylim(min(y_series.min(), predictions_test.min()) - 10, max(y_series.max(), predictions_test.max()) + 10) 
if not os.path.exists('static'):
    os.makedirs('static')
plt.savefig('static/prediccion_precipitacion_P43_33_ml_test_optimized.png')
plt.close()
print("Gráfico de predicciones del modelo ML optimizado en conjunto de prueba guardado como 'static/prediccion_precipitacion_P43_33_ml_test_optimized.png'")


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
            # Fallback si el buffer es muy pequeño, aunque con MAX_LAG_NEEDED no debería pasar si hay suficientes datos históricos
            current_features_dict[f'target_lag_{lag_val}'] = y_series.mean() 

    # Estadísticas de ventana móvil (usando el history_buffer)
    temp_series = pd.Series(list(history_buffer)) 
    for window in Rolling_windows_precipitacion:
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
    future_features_df_single_row = future_features_df_single_row.reindex(columns=X_train.columns, fill_value=0)

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
    'P43_33_Predicho': future_predictions_ml 
})

# --- Guardar en un CSV diferente para las predicciones optimizadas ---
nombre_archivo_salida_ml_futuro_optimizado = f"PreciPredictRF_P43_33_Predicciones_Futuras_Optimizadas_{ANIOS_A_PREDECIR}anios.csv"
final_future_predictions_df.to_csv(nombre_archivo_salida_ml_futuro_optimizado, index=False)
print(f"\nPredicciones futuras del modelo ML (Random Forest Optimizado para P43_33) guardadas en '{nombre_archivo_salida_ml_futuro_optimizado}'")

# --- Visualización de Predicciones Futuras ---
plt.figure(figsize=(16, 8))
plt.plot(y_series.index, y_series, label='Datos Históricos (P43_33)', color='blue')
plt.plot(final_future_predictions_df['Fecha'], final_future_predictions_df['P43_33_Predicho'], 
         label=f'Predicción Futura ML (PreciPredict RF - P43_33 - {ANIOS_A_PREDECIR} años)', color='red', linestyle='--')
plt.title(f'Precipitación Mensual (P43_33): Datos Históricos y Predicción Futura con PreciPredict RF')
plt.xlabel('Fecha')
plt.ylabel('Precipitación P43_33 (mm)')
plt.legend()
plt.grid(True)
plt.tick_params(axis='x', rotation=45)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig('static/prediccion_precipitacion_P43_33_ml_future_optimized.png')
plt.close()
print("Gráfico de predicciones futuras del modelo ML optimizado guardado como 'static/prediccion_precipitacion_P43_33_ml_future_optimized.png'")