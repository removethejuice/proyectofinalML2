import pandas as pd
import mediapipe as mp
import numpy as np
import os
import cv2
import csv

def get_hand_landmarks(directory, hands, output_csv="hand_landmarks.csv"):
    data = []
    headers = ['image_name']

    # Agregar encabezados para 21 landmarks (x, y, z) => 63 columnas
    for i in range(21):
        headers += [f'x_{i}', f'y_{i}', f'z_{i}']

    path = "hand_landmarks"
    split_dir = os.path.join(path, directory)
    for filename in os.listdir(split_dir):
        if filename.endswith('.jpg'):
            print(f"Processing {filename} in {directory}")
            img_path = os.path.join(split_dir, filename)
            image = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                print(f"Found {len(results.multi_hand_landmarks)} hands in {filename}")
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [filename]
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    data.append(landmarks)

    # Guardar en CSV
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    
    print(f"CSV guardado como {output_csv}")

def get_y_values(filepath, output_csv):
    input_csv = f"hand_landmarks\\{filepath}\\_classes.csv"
    df = pd.read_csv(input_csv)
    df = df.sort_values("filename").reset_index(drop=True)
    class_columns = df.columns[1:]
    def get_y_value(row):
        for col in class_columns:
            if row[col] == 1 and len(col) !=2:
                return col
        return None  
    df["y_value"] = df.apply(get_y_value, axis=1)
    df[["filename", "y_value"]].to_csv(output_csv, index=False)
    print(f"Archivo 'y_values_{filepath}.csv' generado correctamente.")

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )

    get_hand_landmarks('test', hands, "landmarks_test.csv")
    get_hand_landmarks('train', hands, "landmarks_train.csv")
    get_hand_landmarks('valid', hands, "landmarks_valid.csv")

    
    get_y_values('test', "y_values_test.csv")
    get_y_values('train', "y_values_train.csv")
    get_y_values('valid', "y_values_valid.csv")

    hands.close()
    mp_hands.solutions.drawing_utils.DrawingSpec().close()

if __name__ == '__main__':
    main()



   
    
