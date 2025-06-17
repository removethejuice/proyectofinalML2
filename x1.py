import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cámara no accesible")
else:
    print("✅ Cámara lista")

ret, frame = cap.read()
if ret:
    print("✅ Frame capturado correctamente")
else:
    print("⚠️ No se pudo capturar un frame")
cap.release()
