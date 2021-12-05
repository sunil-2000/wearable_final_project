import serial
import time

########################## initialization steps ################################
ports = ["/dev/cu.usbserial-0246A723","/dev/cu.mask_on-ESP32SPP" ]
arduino_port = ports[1]  # serial port of Arduino

baud = 115200
cur_time = time.localtime()
ser = serial.Serial(arduino_port, baud)

i = 0
while i < 10:
  ser.write(i%2)
  time.sleep(3)
  i += 1 
