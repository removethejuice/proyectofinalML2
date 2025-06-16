# REQUIREMENTS:
# pip install mediapipe opencv-python scikit-learn joblib

import cv2
import mediapipe as mp
import joblib
import numpy as np

# Cargar modelo y scaler
mlp = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # una mano por predicción
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

# Captura de video
video_capture = cv2.VideoCapture(0)

while True:
    success, image = video_capture.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    prediction_label = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas x, y, z de los 21 puntos
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Asegurar que tiene 63 features
            if len(landmarks) == 63:
                landmarks_np = np.array(landmarks).reshape(1, -1)
                landmarks_scaled = scaler.transform(landmarks_np)
                prediction = mlp.predict(landmarks_scaled)
                prediction_label = prediction[0]

    # Mostrar resultado en pantalla
    if prediction_label:
        cv2.putText(image, f"Prediccion: {prediction_label}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Hands - Predicción', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

video_capture.release()
cv2.destroyAllWindows()
