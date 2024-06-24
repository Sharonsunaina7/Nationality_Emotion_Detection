import tkinter as tk
from tkinter import filedialog
from tkinter import *
from sklearn import metrics
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def NationalityModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def AgeModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
def detect_dress_color(image, face_coords):
    x, y, w, h = face_coords
    dress_region = image[y + h:y + 2 * h, x:x + w]

    dress_region = cv2.cvtColor(dress_region, cv2.COLOR_BGR2RGB)
    dress_region = cv2.resize(dress_region, (50, 50))

    pixels = np.float32(dress_region.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels , palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    dominant_color = palette[0].astype(int)
    return tuple(dominant_color)

def detect_dress_color(image, face_coords):
    x, y, w, h = face_coords
    dress_region = image[y + h:y + 2 * h, x:x + w]

    dress_region = cv2.cvtColor(dress_region, cv2.COLOR_BGR2RGB)
    dress_region = cv2.resize(dress_region, (50, 50))

    pixels = np.float32(dress_region.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    dominant_color = palette[0].astype(int)

    color_ranges = {
        "Red": [(150, 0, 0), (255, 100, 100)],
        "Green": [(0, 100, 0), (100, 255, 100)],
        "Blue": [(0, 0, 150), (100, 100, 255)],
        "Yellow": [(225, 225, 0), (255, 255, 150)],
        "Orange": [(255, 165, 0), (255, 200, 100)],
        "Pink": [(255, 182, 193), (255, 192, 203)],
        "Purple": [(128, 0, 128), (150, 50, 150)],
        "Brown": [(101, 67, 33), (150, 100, 50)],
        "Gray": [(100, 100, 100), (200, 200, 200)],
        "Black": [(0, 0, 0), (50, 50, 50)],
        "White": [(200, 200, 200), (255, 255, 255)]
        }

    for color_name, (lower, upper) in color_ranges.items():
        if all(lower[i] <= dominant_color[i] <= upper[i] for i in range(3)):
            return color_name

    return "Vibrant Color"

top = tk.Tk()
top.geometry('1100x800')
top.title('Nationality and Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label2 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label3 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label4 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nationality_model = NationalityModel("model_nationality8.json", "nationality_detect7.weights.h5")
emotion_model = FacialExpressionModel("model_a1.json", "model_weights1.weights.h5")
age_model = AgeModel("model_age3.json", "model_age.weights.h5")

EMOTION_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NATIONALITY_LIST = ["Indian", "American", "African", "Asian", "Others"]
AGE_LIST = ["1-10", "11-19", "20-29", "30-39", "40-49", "50-59", "60-110"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)

    try:
        for (x, y, w, h) in faces:
            fc_age = image[y:y+h, x:x+w]
            roi_age = cv2.resize(fc_age, (200, 200))

            age_pred = AGE_LIST[np.argmax(age_model.predict(roi_age[np.newaxis,:,:,:]))]
            if age_pred == "1-10" or age_pred == "60-110":
                label3.configure(foreground="#011638", text="Age: Upload image between 10-60 years old")
                label1.configure(text="")
                label2.configure(text="")
                label4.configure(text="")
                return

            label3.configure(foreground="#011638", text="Age range between: " + str(age_pred))

            fc_emotion = gray_image[y:y+h, x:x+w]
            roi_emotion = cv2.resize(fc_emotion, (48, 48))

            fc_nationality = image[y:y+h, x:x+w]
            roi_nationality = cv2.resize(fc_nationality, (200, 200))

            emotion_pred = EMOTION_LIST[np.argmax(emotion_model.predict(roi_emotion[np.newaxis, :, :, np.newaxis]))]

            nationality_pred = NATIONALITY_LIST[np.argmax(nationality_model.predict(roi_nationality[np.newaxis,:,:,:]))]
            label1.configure(foreground="#011638", text="Emotion: " + emotion_pred)
            label2.configure(foreground="#011638", text="Nationality: " + nationality_pred)

            if nationality_pred in ["Indian", "African"]:

                dress_color = detect_dress_color(image, (x, y, w, h))
                dress_color_text = f"Dress Color: RGB{dress_color}"
                label4.configure(foreground="#011638", text=dress_color_text)
            else:
                label4.configure(text="")

    except Exception as e:
        print(e)
        label1.configure(foreground="#011638", text="Unable to detect")
        label2.configure(foreground="#011638", text="")
        label3.configure(foreground="#011638", text="")
        label4.configure(foreground="#011638", text="")

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Nationality,Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        label2.configure(text='')
        label3.configure(text='')
        label4.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        label1.configure(foreground="#011638", text="Error: " + str(e))

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
label2.pack(side='bottom', expand='True')
label3.pack(side='bottom', expand='True')
label4.pack(side='bottom', expand='True')
heading = Label(top, text='Nationality and Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()