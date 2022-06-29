import serial
import time

########################## initialization steps ################################
ports = ["/dev/cu.usbserial-0246A723", "/dev/cu.maskon-ESP32SPP"]
arduino_port = ports[1]  # serial port of Arduino

baud = 115200
cur_time = time.localtime()
csv_file_name = input("csv_file_name:\n")
print("Label +1 = improper wear, Lable -1 = proper wear")
labels = int(input("enter label:\n"))
n = input("trial number:\n")
file_name = "{}.csv".format(csv_file_name)
ser = serial.Serial(arduino_port, baud)
print("Connected to Arduino port:" + arduino_port)
file = open('./facial_try2/'+file_name, "a")
print("Created file")
################################################################################
samples, line = 80, 0
while line <= samples:
    getData = str(ser.readline())
    data = getData[0:][2:-4]+","+str(labels)
    print("line: {}, data: {}".format(line, data))
    file.write(data +n+ "\n")  # write data with a newline
    print('collected:{} samples'.format(line))
    line += 1

# close out the file
file.close()
