import serial
import time
import joblib
import numpy as np
from feature_engineering import Features

########################## initialization steps ################################
ports = ["/dev/cu.usbserial-0246A723", "/dev/cu.maskon-ESP32SPP"]
arduino_port = ports[1]  # serial port of Arduino

baud = 115200
cur_time = time.localtime()
ser = serial.Serial(arduino_port, baud)

# import model
trained_classifier = joblib.load('my_model.pkl')

data = []
rt = 0
dim = 4
predictions = []

while True:
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
    if(len(data) == 100):
        # create feature test point
        data_matrix = np.array(data)
        window_data = Features.gen_features_test(data_matrix)
        pred = trained_classifier.predict(window_data)

        predictions.append(pred)
       # labels +1 / -1
        if pred == 1:
            print("improper wear, pred= {}".format(pred))
            str_pred = str(pred)
            byte_pred = str.encode(str_pred)
            ser.write(byte_pred)
        else:
            print("proper wear, pred = {}".format(pred))
        rt = 50
        data = data[50:]
        predictions.append(pred)
    rt += 1
