import pandas as pd

# Ruta relativa del archivo original
input_csv = "LESHO prueba.v1i.multiclass\\test\\_classes.csv"

# Cargar el archivo
df = pd.read_csv(input_csv)

# Ordenar alfabéticamente por nombre de archivo
df = df.sort_values("filename").reset_index(drop=True)

# Identificar las columnas de clases (todas excepto 'filename')
class_columns = df.columns[1:]

# Función para extraer la clase (letra) de cada fila
def get_y_value(row):
    for col in class_columns:
        if row[col] == 1:
            return col
    return None  # en caso de error

# Aplicar la función a cada fila
df["y_value"] = df.apply(get_y_value, axis=1)

# Guardar el resultado en y_values.csv
df[["filename", "y_value"]].to_csv("y_values.csv", index=False)

print("✅ Archivo 'y_values.csv' generado correctamente.")
