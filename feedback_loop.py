import serial
import time
import joblib
from feature_engineering import Features 

########################## initialization steps ################################
ports = ["/dev/cu.usbserial-0246A723","/dev/cu.mask_on-ESP32SPP" ]
arduino_port = ports[1]  # serial port of Arduino

baud = 115200
cur_time = time.localtime()
ser = serial.Serial(arduino_port, baud)

# import model
trained_classifier = joblib.load('my_model.pkl')

i = 0
data = []
while i < 10:
  ser.write(i%2)
  time.sleep(3)
  i += 1 
  line = str(ser.readline())
  ### first convert data in sliding window to feature data point
  datum = line[0:][2:-4]
  datum = [x.strip() for x in datum.split(',')]
  data.append(datum)

  if len(data) > 100:
    # process data 
    ### apply model to data point
    window_data = Features.gen_features_test(data)
    pred = trained_classifier.predict(data)
    # labels +1 / -1 
    output = 1 if pred == 1 else 0
    ser.write(output)
    ### send prediction to microprocessor 
