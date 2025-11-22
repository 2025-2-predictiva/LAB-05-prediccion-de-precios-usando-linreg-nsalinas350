#
# En este dataset se desea pronosticar el precio de vehiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#

import pandas as pd
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import os
from glob import glob 

# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#

def load_data():
    DATA_PATH = "files/input/"
    df_train = pd.read_csv(DATA_PATH +  "train_data.csv.zip")
    df_test = pd.read_csv(DATA_PATH + "test_data.csv.zip")
    return df_train, df_test

def clean_data(df):
    df_copy = df.copy()
    current_year = 2021
    columns_to_drop = ['Year', 'Car_Name']
    df_copy["Age"] = current_year - df_copy["Year"]
    df_copy = df_copy.drop(columns=columns_to_drop)
    
    return df_copy

df_train, df_test = load_data()
df_train = clean_data(df_train)
df_test = clean_data(df_test)

#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#

# --- 1. División del Dataset de Entrenamiento ---
x_train = df_train.drop(columns=['Present_Price'])
y_train = df_train['Present_Price']

# --- 2. División del Dataset de Prueba ---
x_test = df_test.drop(columns=['Present_Price'])
y_test = df_test['Present_Price']

#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#

# 1. Definir las columnas
categorical_features=['Fuel_Type','Selling_type','Transmission']
numerical_features= [col for col in x_train.columns if col not in categorical_features]

# 2. Crear el transformador de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        # Aplicar OneHotEncoder a las variables categóricas
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('scaler',MinMaxScaler(),numerical_features),
    ],
    # Si alguna columna no está en las listas, la eliminamos
    remainder='drop' 
)

# 3. Crear el Pipeline completo
# El Pipeline secuencia el preprocesamiento y el modelo.
def create_pipeline():
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('feature_selection',SelectKBest(f_regression)),
        ('classifier', LinearRegression())
    ])
    return pipeline

model_pipeline = create_pipeline()

#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#

def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
    'feature_selection__k':range(1,25),
    'classifier__fit_intercept':[True,False],
    'classifier__positive':[True,False]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(x_train, y_train)

    # Mostrar los mejores resultados
    print("\n--- 🏆 Resultados de la Optimización ---")
    print(f"Mejor score (Precisión Balanceada): {grid_search.best_score_}")
    print(f"Mejores Hiperparámetros: {grid_search.best_params_}")
    return grid_search

# Guardar el pipeline optimizado
best_model_pipeline = optimize_hyperparameters(model_pipeline, x_train, y_train)

#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#

# Nombre del archivo de destino
MODEL_PATH = "files/models/model.pkl.gz"


model_dir = os.path.dirname(MODEL_PATH)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)

    print(f"Directorio creado: {model_dir}")
with gzip.open(MODEL_PATH, 'wb') as f: # 'wb' = write binary
    pickle.dump(best_model_pipeline, f) 

#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
#

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calcula y formatea las métricas solicitadas."""
    return {
        "type": "metrics",
        "dataset": dataset_name,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(median_absolute_error(y_true, y_pred))
    }

# --- 1. Hacer Predicciones ---
y_pred_train = best_model_pipeline.predict(x_train)
y_pred_test = best_model_pipeline.predict(x_test)

# --- 2. Calcular Métricas ---
metrics_train = calculate_metrics(y_train, y_pred_train, 'train')
metrics_test = calculate_metrics(y_test, y_pred_test, 'test')

results_list = [metrics_train, metrics_test]


# --- 3. Guardar las Métricas en formato JSON Lines ---
OUTPUT_PATH = "files/output/metrics.json" 

# Crear la carpeta de destino si no existe
output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Usar un bucle para escribir cada objeto JSON en una línea separada.
with open(OUTPUT_PATH, 'w') as f:
    for item in results_list:
        f.write(json.dumps(item) + '\n')
