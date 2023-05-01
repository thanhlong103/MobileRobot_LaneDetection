//**********6 Channels L298N Motor Driver**********//
#define ENA A0
#define ENB A1
#define IN1 6
#define IN2 9
#define IN3 10
#define IN4 11

int speed = 180;
int set_speed = 80;
int s = 10;
char command;

//---------------------------FUNCTIONS---------------------------//
void forward()
{
  analogWrite(IN1, 0);
	analogWrite(IN2, set_speed);
  analogWrite(IN3, set_speed);
	analogWrite(IN4, 0);
}

void backward()
{
  analogWrite(IN1, set_speed);
	analogWrite(IN2, 0);
  analogWrite(IN3, 0);
	analogWrite(IN4, set_speed);
}

void right()
{
  analogWrite(IN1, 0);
	analogWrite(IN2, set_speed);
  analogWrite(IN3, 0);
	analogWrite(IN4, set_speed);
}

void left()
{
  analogWrite(IN1, set_speed);
	analogWrite(IN2, 0);
  analogWrite(IN3, set_speed);
	analogWrite(IN4, 0);
}

void stop()
{
  analogWrite(IN1, 0);
	analogWrite(IN2, 0);
  analogWrite(IN3, 0);
	analogWrite(IN4, 0);
}

void setup() {
  Serial.begin(9600);

	pinMode(ENA, OUTPUT);
	pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
	pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
	pinMode(IN4, OUTPUT);

  // Turn off motors - Initial state
  analogWrite(IN1, 0);
	analogWrite(IN2, 0);
  analogWrite(IN3, 0);
	analogWrite(IN4, 0);

  //-----Connect to Raspberry Pi-----//
  // while (!Serial);
  // Serial.println("Opencv Lane Detect Autonomous Robot")
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0)
  {
    analogWrite(ENA, speed);
    analogWrite(ENB, speed);
    command = Serial.read();
    Serial.print("Command: ");
    Serial.println(command);
    switch(command)
    {
      case 'a':
        forward();
        // delay(10);
        // stop();
        break;
      case 'b':
        analogWrite(IN1, 0);
        analogWrite(IN2, set_speed);
        analogWrite(IN3, set_speed + s);
        analogWrite(IN4, 0);
        break;
      case 'c':
        analogWrite(IN1, 0);
        analogWrite(IN2, set_speed);
        analogWrite(IN3, set_speed + s*2);
        analogWrite(IN4, 0);
        break;
      case 'd':
        analogWrite(IN1, 0);
        analogWrite(IN2, set_speed + s);
        analogWrite(IN3, set_speed);
        analogWrite(IN4, 0);
        break;
      case 'e':
        analogWrite(IN1, 0);
        analogWrite(IN2, set_speed + s*2);
        analogWrite(IN3, set_speed);
        analogWrite(IN4, 0);
        break;
      case 'f':
        analogWrite(IN1, 0);
        analogWrite(IN2, set_speed);
        analogWrite(IN3, set_speed + s*3);
        analogWrite(IN4, 0);
        break;
      case 'g':
        analogWrite(IN1, 0);
        analogWrite(IN2, set_speed + s*3);
        analogWrite(IN3, set_speed);
        analogWrite(IN4, 0);
        break;                  
    }
  }
}

