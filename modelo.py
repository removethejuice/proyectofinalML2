import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
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

# Hacer merge por image_name (X) y filename (y)
train = pd.merge(X_train, y_train, left_on='image_name', right_on='filename')
valid = pd.merge(X_valid, y_valid, left_on='image_name', right_on='filename')
test  = pd.merge(X_test,  y_test,  left_on='image_name', right_on='filename')

# Verificar dimensiones
print("Train:", train.shape)
print("Valid:", valid.shape)
print("Test:", test.shape)

# Eliminar columnas innecesarias
train = train.drop(columns=['image_name', 'filename'])
valid = valid.drop(columns=['image_name', 'filename'])
test  = test.drop(columns=['image_name', 'filename'])

train.dropna(inplace=True)
valid.dropna(inplace=True) 
test.dropna(inplace=True)
# Separar X e y finales
X_train = train.drop(columns=['y_value'])
y_train = train['y_value']

X_valid = valid.drop(columns=['y_value'])
y_valid = valid['y_value']

X_test = test.drop(columns=['y_value'])
y_test = test['y_value']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# Evaluar en validación
y_valid_pred = mlp.predict(X_valid_scaled)
print("VALIDACIÓN:")
print(classification_report(y_valid, y_valid_pred))

# Evaluar en test
y_test_pred = mlp.predict(X_test_scaled)
print("TEST:")
print(classification_report(y_test, y_test_pred))

# Guardar el modelo MLP
joblib.dump(mlp, "mlp_model.pkl")

# Guardar el scaler
joblib.dump(scaler, "scaler.pkl")

print("Modelo y scaler guardados exitosamente.")