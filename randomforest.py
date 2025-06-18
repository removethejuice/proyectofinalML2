import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Función para cargar y preparar características
def load_and_prepare_features(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='image_name').reset_index(drop=True)
    return df

# Función para cargar y preparar etiquetas
def load_and_prepare_labels(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='filename').reset_index(drop=True)
    return df

# Cargar características
X_train = load_and_prepare_features("landmarks_train.csv")
X_valid = load_and_prepare_features("landmarks_valid.csv")
X_test  = load_and_prepare_features("landmarks_test.csv")

# Cargar etiquetas
y_train = load_and_prepare_labels("y_values_train.csv")
y_valid = load_and_prepare_labels("y_values_valid.csv")
y_test  = load_and_prepare_labels("y_values_test.csv")

# Hacer merge por nombre
train = pd.merge(X_train, y_train, left_on='image_name', right_on='filename')
valid = pd.merge(X_valid, y_valid, left_on='image_name', right_on='filename')
test  = pd.merge(X_test,  y_test,  left_on='image_name', right_on='filename')

print("Train:", train.shape)
print("Valid:", valid.shape)
print("Test:", test.shape)

# Limpiar columnas innecesarias
train = train.drop(columns=['image_name', 'filename'])
valid = valid.drop(columns=['image_name', 'filename'])
test  = test.drop(columns=['image_name', 'filename'])

train.dropna(inplace=True)
valid.dropna(inplace=True)
test.dropna(inplace=True)

# Separar características y etiquetas
X_train = train.drop(columns=['y_value'])
y_train = train['y_value']
X_valid = valid.drop(columns=['y_value'])
y_valid = valid['y_value']
X_test = test.drop(columns=['y_value'])
y_test = test['y_value']

# Normalizar 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Combinar train + valid para grid search con cross validation
X_search = pd.concat([pd.DataFrame(X_train_scaled), pd.DataFrame(X_valid_scaled)], ignore_index=True)
y_search = pd.concat([y_train, y_valid], ignore_index=True)

# Configurar la grid del grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

print(" Buscando mejores hiperparámetros con GridSearchCV...")

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_search, y_search)

best_model = grid_search.best_estimator_
print("Mejor configuracion encontrada:", grid_search.best_params_)

# Evaluar en test
y_test_pred = best_model.predict(X_test_scaled)
print("\n TEST:")
print(classification_report(y_test, y_test_pred))

# Guardar modelo y scaler
joblib.dump(best_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(" Modelo y scaler guardados exitosamente.")
