#include <CapacitiveSensor.h> 
#include "BluetoothSerial.h"

CapacitiveSensor nose = CapacitiveSensor(26,25); // 1-10M resistor between pins 0 & 4, pin 4 is sensor pin 
CapacitiveSensor chin = CapacitiveSensor(12,27);
CapacitiveSensor leftCheek = CapacitiveSensor(33,15);
CapacitiveSensor rightCheek = CapacitiveSensor(32,14);
BluetoothSerial SerialBT;

void setup() 
{ 
Serial.begin(115200);
SerialBT.begin("mask_on");
pinMode(21, OUTPUT);

} 
void loop() 
{ 
  long nose_capacitance = nose.capacitiveSensor(20); //  number of Samples. The more samples it measured, the longer time it takes. 
  long chin_capacitance = chin.capacitiveSensor(20); //  number of Samples. The more samples it measured, the longer time it takes. 
  long left_capacitance = leftCheek.capacitiveSensor(20); //  number of Samples. The more samples it measured, the longer time it takes. 
  long right_capacitance = rightCheek.capacitiveSensor(20); //  number of Samples. The more samples it measured, the longer time it takes. 
  SerialBT.print(nose_capacitance); // print sensor output 
  SerialBT.print(","); // print sensor output 
  SerialBT.print(chin_capacitance); // print sensor output
  SerialBT.print(","); // print sensor output 
  SerialBT.print(left_capacitance); // print sensor output 
  SerialBT.print(","); // print sensor output 
  SerialBT.print(right_capacitance); // print sensor output  
  SerialBT.print("\n");
  if(SerialBT.available()){
    int c = SerialBT.read();
    char output = char(c);
    if(output == '1') {
        digitalWrite(21, HIGH); 
        delay(1000);
        digitalWrite(21, LOW);
      }
    }
  delay(20); // arbitrary delay to limit data to serial port 
}