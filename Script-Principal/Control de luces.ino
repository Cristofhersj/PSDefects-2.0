const int signalPin = 9;  // Replace with the actual pin connected to your load (vision light, for example)
const int signalPin2 = 10;
const int squareWavePin = 6;  // Replace with the actual pin for the square wave
const int triggerPin = 11;    // Define pin 11 for special trigger

bool cyclesEnabled = false;
unsigned long previousMillis = 0;
unsigned long interval = 1;  // Adjust the interval for PIN 9 as needed

bool triggerActive = false;
unsigned long triggerMillis = 0;
unsigned long triggerDuration = 500;

void setup() {
  pinMode(signalPin, OUTPUT);
  pinMode(squareWavePin, OUTPUT);
  pinMode(signalPin2, OUTPUT);
  pinMode(triggerPin, OUTPUT);  // Set pin 11 as an output
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();

    if (receivedChar == 'F') {
      startCycles();
    } else if (receivedChar == 'P') {
      stopCycles();
    } else if (receivedChar == 'G') {
      activateTrigger();
    }
  }

  if (cyclesEnabled) {
    squareWaveGenerator();
    if (millis() - previousMillis >= interval) {
      previousMillis = millis();
      digitalWrite(signalPin, HIGH);
      digitalWrite(signalPin2, HIGH);
    } else {
      digitalWrite(signalPin, LOW);
      digitalWrite(signalPin2, LOW);
    }
  }

  if (triggerActive && millis() - triggerMillis >= triggerDuration) {
    digitalWrite(triggerPin, LOW);
    triggerActive = false;  // Reset trigger state
  }
}

void startCycles() {
  cyclesEnabled = true;
}

void stopCycles() {
  cyclesEnabled = false;
  digitalWrite(signalPin, LOW);  // Ensure the load is off when cycles are stopped
  digitalWrite(signalPin2, LOW);
}

void squareWaveGenerator() {
  static unsigned long squareWaveMillis = 0;
  static int state = LOW;

  if (millis() - squareWaveMillis >= 10) {
    squareWaveMillis = millis();  // Save the last time the square wave state was updated
    state = !state;
    digitalWrite(squareWavePin, state);
  }
}

void activateTrigger() {
  if (!triggerActive) {  // Ensure we only activate if not already active
    digitalWrite(triggerPin, HIGH);
    triggerMillis = millis();
    triggerActive = true;
  }
}
