import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import time

# Cargar modelo y scaler, aca literal solo cargamos los modelos
mlp = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

# Variables de control, estas variables determinan cuando se envia los frames al modelo
ultimo_registro = None
frames_desde_ultimo = 0
cooldown_frames = 20
current_max_hands = 2

# GUI setup esta parte me daba error
root = tk.Tk()
root.title("Reconocimiento de señas")
root.geometry("900x700")

max_hands_var = tk.IntVar(value=current_max_hands)

# Captura de video esta parte debe ir antes de crear la GUI
cap = cv2.VideoCapture(0)
time.sleep(1)
if not cap.isOpened():
    print("No se pudo abrir la camara!!!!")
    exit()

# Widgets de GUI, esta parte de GUI me daba error asi que Chatsito me quito los errores odio programar GUI gracias a dios no estudie sistemas porque no podria sobrevivir un trabajo en frontend preferirira ser un campesino medieval
video_label = tk.Label(root)
video_label.pack(pady=10)

config_frame = ttk.Frame(root)
config_frame.pack()
ttk.Label(config_frame, text="Máx. manos:").pack(side="left")
ttk.Spinbox(config_frame, from_=1, to=2, textvariable=max_hands_var, width=5).pack(side="left")

text_frame = ttk.Frame(root)
text_frame.pack(pady=10, fill="both", expand=True)
scrollbar = ttk.Scrollbar(text_frame)
scrollbar.pack(side="right", fill="y")
text_box = tk.Text(text_frame, height=8, font=("Arial", 14), wrap="word", yscrollcommand=scrollbar.set)
text_box.pack(side="left", fill="both", expand=True)
scrollbar.config(command=text_box.yview)

def guardar():
    texto = text_box.get("1.0", tk.END)
    ruta = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texto", "*.txt")])
    if ruta:
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(texto.strip())

def borrar():
    text_box.delete("1.0", tk.END)

btn_frame = ttk.Frame(root)
btn_frame.pack(pady=10)
ttk.Button(btn_frame, text="Guardar texto", command=guardar).pack(side="left", padx=5)
ttk.Button(btn_frame, text="Borrar texto", command=borrar).pack(side="left", padx=5)

# MediaPipe setup, esto es lo que estaba antes, solo agregamos el GUI encima de esto
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=max_hands_var.get(),# este codigo se encarga de poner el maximo de manos que se pueden detectar
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

# Loop de video, se me habia olvidado que habia que poner todo en un loop para que se actualice la imagen de la camara y se procese el modelo, esto es lo que hace que la camara funcione en tiempo real
def update():
    global hands, ultimo_registro, frames_desde_ultimo, current_max_hands

    try:
        # Si cambia el numero de manos se reinicia 
        if max_hands_var.get() != current_max_hands:
            current_max_hands = max_hands_var.get()
            hands.close()
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=current_max_hands,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )

        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar frame")
            root.after(10, update)
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        pred = ""

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                coords = []
                for lm in hand.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                if len(coords) == 63:
                    arr = np.array(coords).reshape(1, -1)
                    scaled = scaler.transform(arr)
                    pred = mlp.predict(scaled)[0]

        if pred:
            cv2.putText(frame, f"{pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if pred != ultimo_registro and frames_desde_ultimo >= cooldown_frames:
                text_box.insert(tk.END, pred)
                text_box.see(tk.END)
                ultimo_registro = pred
                frames_desde_ultimo = 0
            else:
                frames_desde_ultimo += 1
        else:
            frames_desde_ultimo += 1

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    except Exception as e:
        print("Error:", e)

    root.after(10, update)

# Inicia la ventana de la GUI y el loop de video
update()
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), hands.close(), root.destroy()))
root.mainloop()
