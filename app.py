import tkinter as tk 
import customtkinter as ck 

import pandas as pd 
import numpy as np 
import pickle 

import mediapipe as mp
import cv2
from PIL import Image, ImageTk 

from landmarks import landmarks

window = tk.Tk()
window.geometry("480x650")
window.title("Arnis") 
ck.set_appearance_mode("dark")

classLabel = ck.CTkLabel(window, height=40, width=150, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=3)
classLabel.configure(text='STRIKE')
probLabel  = ck.CTkLabel(window, height=40, width=100, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=170, y=3)
probLabel.configure(text='PROB') 
counterLabel = ck.CTkLabel(window, height=40, width=150, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=280, y=3)
counterLabel.configure(text='HIGHEST PROB') 
classBox = ck.CTkLabel(window, height=40, width=150, font=("Arial", 20), text_color="white", fg_color="green")
classBox.place(x=10, y=43)
classBox.configure(text='0') 
probBox = ck.CTkLabel(window, height=40, width=100, font=("Arial", 20), text_color="white", fg_color="green")
probBox.place(x=170, y=43)
probBox.configure(text='0') 
counterBox = ck.CTkLabel(window, height=40, width=150, font=("Arial", 20), text_color="white", fg_color="green")
counterBox.place(x=280, y=43)
counterBox.configure(text='0') 

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=100) 
lmain = tk.Label(frame) 
lmain.place(x=0, y=0) 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) 

with open('arnis_svc_model.pkl', 'rb') as file:
    model = pickle.load(file) 

cap = cv2.VideoCapture(0) # adjust on your camera index 0/1
current_stage = ''
highestProb = 0 
bodylang_prob = np.array([0,0]) 
bodylang_class = '' 

def detect(): 
    global current_stage
    global highestProb
    global bodylang_class
    global bodylang_prob 

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius = 5), 
        mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius = 10)) 

    try: 
        row = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[1:]]).flatten().tolist()
        X = pd.DataFrame([row], columns = landmarks) 
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0] 
        max_prob = bodylang_prob[bodylang_prob.argmax()]
        print(bodylang_class, bodylang_prob[bodylang_prob.argmax()])

        if bodylang_class =="1. Right Temple Strike" and max_prob >= 0.5: 
            current_stage = "Right Temple" 
        elif bodylang_class =="2. Stomach Thrust" and max_prob >= 0.45: 
            current_stage = "Stomach Thrust" 
        elif bodylang_class =="3. Left Knee Strike" and max_prob >= 0.5: 
            current_stage = "Left Knee"
        else:
            current_stage = "Strike not clear"

        if max_prob >= .70 and current_stage != "Strike not clear":
            highestProb = bodylang_class.split(". ")[1]

    except Exception as e: 
        print("error: ")
        print(e) 

    img = image[:, :460, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    lmain.after(10, detect)  

    counterBox.configure(text = str(highestProb) + " {:.2f}".format(float(max_prob))) 
    probBox.configure(text = "{:.2f}".format(max_prob)) 
    classBox.configure(text = current_stage) 

detect() 
window.mainloop()