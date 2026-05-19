#include <Wire.h>
#include "Adafruit_VL6180X.h"

Adafruit_VL6180X vl = Adafruit_VL6180X();

int seq_num = 0;

void setup() {
  Serial.begin(115200);

  // wait for serial port to open on native usb devices
  while (!Serial) {
    delay(1);
  }
  
  Serial.println("Adafruit VL6180x test!");
  if (! vl.begin()) {
    Serial.println("Failed to find sensor");
    while (1);
  }
  Serial.println("Sensor found!");
  Serial.println("Start");
  
}

void loop() {
  // float lux = vl.readLux(VL6180X_ALS_GAIN_5);

  // Serial.print("Lux: "); Serial.println(lux);
  
  // measure time to read range
  // uint32_t start = millis();
  uint8_t range = vl.readRange();
  uint8_t status = vl.readRangeStatus();
  // uint32_t stop = millis();
  // Serial.print("Took "); Serial.print(stop - start); Serial.print(" ms to read with status: "); Serial.println(status);

  if(seq_num < 0)
    seq_num = 0;

  seq_num++;
  

  if (status == VL6180X_ERROR_NONE) {
    Serial.print("Range:"); Serial.print(range);
    Serial.print(":Seq:"); Serial.println(seq_num);
  }

  // Some error occurred, print it out!
  
  
  if  ((status >= VL6180X_ERROR_SYSERR_1) && (status <= VL6180X_ERROR_SYSERR_5)) {
    Serial.print("System error");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  else if (status == VL6180X_ERROR_ECEFAIL) {
    Serial.print("ECE failure");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  else if (status == VL6180X_ERROR_NOCONVERGE) {
    Serial.print("No convergence");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  else if (status == VL6180X_ERROR_RANGEIGNORE) {
    Serial.print("Ignoring range");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  else if (status == VL6180X_ERROR_SNR) {
    Serial.print("Signal/Noise error");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  else if (status == VL6180X_ERROR_RAWUFLOW) {
    Serial.print("Raw reading underflow");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  else if (status == VL6180X_ERROR_RAWOFLOW) {
    Serial.print("Raw reading overflow");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  else if (status == VL6180X_ERROR_RANGEUFLOW) {
    Serial.print("Range reading underflow");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  else if (status == VL6180X_ERROR_RANGEOFLOW) {
    Serial.print("Range reading overflow");     Serial.print(":Seq:"); Serial.println(seq_num);

  }
  delay(20);
}