#REQUIREMENTS:
# pip install mediapipe opencv-python
# NEEDS PYTHON 3.8

import cv2
import mediapipe as mp

video_capture = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mp_drawing = mp.solutions.drawing_utils

while True:
    success, image = video_capture.read()

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow('MediaPipe Hands', image)
    cv2.waitKey(1)