import serial
import time
import joblib
import numpy as np
from feature_engineering import Features
from tkinter import*
from PIL import Image, ImageTk

########################## initialization steps ################################
ports = ["/dev/cu.usbserial-0246A723", "/dev/cu.maskon-ESP32SPP"]
arduino_port = ports[1]  # serial port of Arduino

baud = 115200
cur_time = time.localtime()
ser = serial.Serial(arduino_port, baud)


def change_img(pred, face_pred):
   if(pred == 1):
    img = img1
    second_img = img6
   else:
    img = img2
    if face_pred == -1:
        second_img_img = img6
    if(face_pred == 0):
        second_img = img5
    if(face_pred == 1):
        second_img = img4
    if(face_pred == 2):
        second_img = img3
    if(face_pred == 3):
        second_img = img3
   label.configure(image=img)
   label.image=img
   label2.configure(image=second_img)
   label2.image=second_img


win= Tk()
win.geometry("750x600")
win.title("Gallery")

img1= ImageTk.PhotoImage(Image.open("./image/cancel.png"))
img2=ImageTk.PhotoImage(Image.open("./image/checkmark.jpg"))
img3= ImageTk.PhotoImage(Image.open("./image/chew_talk.jpg"))
img4=ImageTk.PhotoImage(Image.open("./image/masked.png"))
img5= ImageTk.PhotoImage(Image.open("./image/smile.png"))
img6=ImageTk.PhotoImage(Image.open("./image/unmasked.png"))

label= Label(win,image= img1)
label.pack(side=LEFT)
label2 = Label(win,image= img1)
label2.pack(side=RIGHT)
win.bind("<Return>", change_img)

# import model
binary_classifier = joblib.load('my_model_binary.pkl')
facial_classifier = joblib.load('my_model_facial.pkl')


def fd_loop():
    data = []
    rt = 0
    dim = 4
    predictions = []
    window_size = 80
    overlap = 0.5
    n = 0
    while True:
        win.update_idletasks()
        win.update()
        line = str(ser.readline())
        datum = line[0:][2:-4]
        datum = [x for x in datum.split(',')]

        for i in range(len(datum)):
            d = datum[i]
            if d == '':
                datum[i] = 0.
            else:
                datum[i] = float(d)

        data.append(datum)
        if(len(data) == window_size):
            print(n)
            n+=1
            # create feature test point
            data_matrix = np.array(data)
            window_data = Features.gen_features_test(data_matrix)
            pred = binary_classifier.predict(window_data)

            predictions.append(pred)
        # labels +1 / -1
            face_pred = -1
            
            if pred == 1:
                print("improper wear, pred= {}".format(pred))
                str_pred = str(pred)
                byte_pred = str.encode(str_pred)
                ser.write(byte_pred)
            else:
                face_pred = facial_classifier.predict(window_data)
                print("proper wear, pred = {}".format(pred))
                if face_pred == 0:
                    print("\nsmile, pred = {}".format(face_pred))
                if face_pred == 1:
                    print("\nneutral, pred = {}".format(face_pred))
                if face_pred == 2:
                    print("\nchewing, pred = {}".format(face_pred))
                if face_pred == 3:
                    print("\ntalking, pred = {}".format(face_pred))
            change_img(pred, face_pred)
            rt = int(window_size * overlap)
            data = data[rt:]
            predictions.append(pred)
        rt += 1

if __name__ == '__main__':
    fd_loop()
