#include <Servo.h>

Servo myservo;

int pos_ini = 90;  // Start position
int max_pos = 70;  
void setup() {
  myservo.attach(9);
  Serial.begin(9600);
  myservo.write(pos_ini);  // Ensure servo starts in initial position
}

void loop() {
  if (Serial.available() > 0) {
    char received = Serial.read();
    if (received == '1') {
      Cut();
    }
  }
}

void Cut() {
  // Move to max_pos
  for (int pos = pos_ini; pos >= max_pos; pos--) {
    myservo.write(pos);
    delay(20);  // Small delay for smooth motion
  }

  delay(15000);  

  // Move back to pos_ini
  for (int pos = max_pos; pos <= pos_ini; pos++) {
    myservo.write(pos);
    delay(20);  // Small delay for smooth motion
  }
}