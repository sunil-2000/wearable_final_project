import serial
import time

########################## initialization steps ################################
arduino_port = "/dev/cu.usbserial-0246A723"  # serial port of Arduino
baud = 115200
cur_time = time.localtime()
csv_file_name = input("csv_file_name:\n")
labels = int(input("enter label:\n"))

file_name = "{}.csv".format(csv_file_name)

ser = serial.Serial(arduino_port, baud)
print("Connected to Arduino port:" + arduino_port)
file = open(file_name, "a")
print("Created file")
################################################################################
samples, line = 5000, 0
while line <= samples:
    getData = str(ser.readline())
    data = getData[0:][2:-4]+","+str(labels)
    print("line: {}, data: {}".format(line, data))
    file = open(file_name, "a")
    file.write(data + "\n")  # write data with a newline
    line += 1

# close out the file
file.close()
