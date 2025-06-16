import pandas as pd

# Función para cargar y preparar los archivos
def load_and_prepare_features(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='image_name').reset_index(drop=True)
    return df.drop(columns=['image_name'])

def load_and_prepare_labels(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='filename').reset_index(drop=True)
    return df['y_value']  # Asegura que la columna sea llamada 'label'

# Cargar características (X)
X_train = load_and_prepare_features("landmarks_train.csv")
print("XTrain: ", len(X_train))
X_valid = load_and_prepare_features("landmarks_valid.csv")
print("XValid: ", len(X_valid))
X_test  = load_and_prepare_features("landmarks_test.csv")
print("XTest: ", len(X_test))

# Cargar etiquetas (y)
y_train = load_and_prepare_labels("y_values_train.csv")
print("YTrain: ", len(y_train))
y_valid = load_and_prepare_labels("y_values_valid.csv")
print("YValid: ", len(y_valid))
y_test  = load_and_prepare_labels("y_values_test.csv")
print("YTest: ", len(y_test))
