#include <PS2X_lib.h>

// Mega2560 Encoder X 2 Test Program
// Board : 24V 5A MegaShield V1.2
// PS2 조향 분리
#include <SPI.h>
#include <Wire.h>
#include <PS2X_lib.h>  //for v1.6
#include <MsTimer2.h>                 // Timer를 실행하기 위한 라이브러리 추가  

#define chipSelectPin1 22
#define chipSelectPin2 23

#define MOT1E 2
#define MOT1F 3
#define MOT1R 4
#define MOT2E 5
#define MOT2F 6
#define MOT2R 7
#define MOT3E 8
#define MOT3F 9
#define MOT3R 10
#define PS2_DAT        17  //PS.1    
#define PS2_CMD        16  //PS.2
#define PS2_SEL        15  //PS.6
#define PS2_CLK        14  //PS.7

//#define pressures   true
#define pressures   false
//#define rumble      true
#define rumble      false

PS2X ps2x; // create PS2 Controller Class
int error = 0;
byte type = 0;
byte vibrate = 0;

uint32_t oldTime;
//*****************************************************
void setup()
//*****************************************************
{
  Serial.begin(115200);
  Serial1.begin(115200);
  Serial1.println("MS2405 Connected...");
  pinMode(chipSelectPin1, OUTPUT);
  pinMode(chipSelectPin2, OUTPUT);

  digitalWrite(chipSelectPin1, HIGH);
  digitalWrite(chipSelectPin2, HIGH);
  for (int i = 0; i < 5; i++) {
    pinMode(A8 + i, OUTPUT);
    digitalWrite(A8 + i, LOW);
  }
  pinMode(MOT1E, OUTPUT);
  pinMode(MOT1F, OUTPUT);
  pinMode(MOT1R, OUTPUT);
  pinMode(MOT2E, OUTPUT);
  pinMode(MOT2F, OUTPUT);
  pinMode(MOT2R, OUTPUT);
  pinMode(MOT3E, OUTPUT);
  pinMode(MOT3F, OUTPUT);
  pinMode(MOT3R, OUTPUT);
//  pinMode(A0, INPUT_PULLUP);
  digitalWrite(MOT1E, LOW);
  digitalWrite(MOT2E, LOW);
  digitalWrite(MOT3E, LOW);
  digitalWrite(MOT1F, LOW);
  digitalWrite(MOT2F, LOW);
  digitalWrite(MOT3F, LOW);
  digitalWrite(MOT1R, LOW);
  digitalWrite(MOT2R, LOW);
  digitalWrite(MOT3R, LOW);
  Serial.println("MS2405_V1.2 PS2 Control Start");
  delay(300);  //added delay to give wireless ps2 module some time to startup, before configuring it

  LS7366_Init();
  clearEncoderCount(1);
  clearEncoderCount(2);

  error = ps2x.config_gamepad(PS2_CLK, PS2_CMD, PS2_SEL, PS2_DAT, pressures, rumble);

  oldTime = millis();

  MsTimer2::set(50, CallBack); // 50ms period
//   MsTimer2::set(50, CallBack); // 50ms period
  MsTimer2::start();
}

void CallBack() {
  PS2_Recv();
}

int steer=0, spd=0;
bool Brake;
void PS2_Recv() {
  ps2x.read_gamepad(false, vibrate); //read controller and set large motor to spin at 'vibrate' speed
  steer = map(ps2x.Analog(PSS_LX), 0, 255, -250, 250);
  spd = map(ps2x.Analog(PSS_RY), 0, 255, -250, 250);
  //  PS2_key();
}

bool dir;
void loop()
{
  if ((millis() - oldTime) > 500) {
    Report();
    //    CHK = 25;
    Serial.print("PWM Value :  ");
    Serial.print(steer);
    Serial.print(" /  " );
    Serial.println(spd);
    Serial.print("Angle: ");
    Serial.println(analogRead(A15));
  }

  if(abs(spd) < 256) {
    spd > 0 ? dir = 0 : dir = 1;
    abs(spd) > 10 ? spd : spd = 0;
  }

  MOT_DRV(dir, spd);
  steer_control(steer);
  EncoderCHK();
}

void MOT_DRV(bool a, uint8_t b) {
  digitalWrite(MOT1F, dir);
  digitalWrite(MOT1R, !dir);
  digitalWrite(MOT2F, dir);
  digitalWrite(MOT2R, !dir);
  analogWrite(MOT1E, abs(spd));
  analogWrite(MOT2E, abs(spd));
}

//*****************************************************
long getEncoderValue(int encoder)
//*****************************************************
{
  byte count1Value, count2Value, count3Value, count4Value;
  long result;

  digitalWrite(chipSelectPin1 + encoder - 1, LOW);

  SPI.transfer(0x60); // Request count
  count1Value = SPI.transfer(0x01); // Read highest order byte
  count2Value = SPI.transfer(0x01);
  count3Value = SPI.transfer(0x01);
  count4Value = SPI.transfer(0x01); // Read lowest order byte

  digitalWrite(chipSelectPin1 + encoder - 1, HIGH);

  result = ((long)count1Value << 24) + ((long)count2Value << 16) + ((long)count3Value << 8) + (long)count4Value;

  return result;
}//end func


void clearEncoderCount(int encoder_no)
{
  digitalWrite(chipSelectPin1 + encoder_no - 1, LOW); // Begin SPI conversation
  SPI.transfer(0xE0);
  digitalWrite(chipSelectPin1 + encoder_no - 1, HIGH); // Terminate SPI conversation
}


// LS7366 Initialization and configuration
//*************************************************
void LS7366_Init(void)
//*************************************************
{
  pinMode(chipSelectPin1, OUTPUT);
  pinMode(chipSelectPin2, OUTPUT);
  digitalWrite(chipSelectPin1, HIGH);
  digitalWrite(chipSelectPin2, HIGH);
  // SPI initialization
  SPI.begin();
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0)); //SPI at 1MHz
  delay(1);

  digitalWrite(chipSelectPin1, LOW);
  SPI.transfer(0x88);
  SPI.transfer(0x02);
  digitalWrite(chipSelectPin1, HIGH);

  digitalWrite(chipSelectPin2, LOW);
  SPI.transfer(0x88);
  SPI.transfer(0x02);
  digitalWrite(chipSelectPin2, HIGH);
}

long OLD_FWD, OLD_Back;
int old;
#define meter 16*20./0.8168 // 16pole / 기어비 20:1 / 타이어지름  26cm -> 원주: 26 x pi
void Report() {
  long New_FWD;
  long New_Back;
  float duty = (millis() - oldTime) / 1000.0;
  //  Serial.println(duty);
  oldTime = millis();
  New_FWD = getEncoderValue(2);
  Serial.print("Encoder Front= ");
  Serial.println(New_FWD);

  New_Back = getEncoderValue(1);
  Serial.print(" Encoder Rear= ");
  Serial.println(New_Back);
  int Angle = analogRead(15);
  if (OLD_FWD != New_FWD || OLD_Back != New_Back || old != Angle ) {
    float SPEED_F = abs(New_FWD - OLD_FWD) / (meter * duty) * 3600  / 1000; //320 = 16 * 20 ; pole * 기어비
    float SPEED_B = abs(New_Back - OLD_Back) / (meter * duty) * 3600  / 1000;
    Serial.print(SPEED_F);
    Serial.print(" km/H  / ");
    Serial.print(SPEED_B);
    Serial.print(" km/H  / ");
    Serial.println(Angle);
    OLD_FWD = New_FWD;
    OLD_Back = New_Back;
  }
}

long oldX, oldY;

void EncoderCHK() {
  long encoder1Value;
  long encoder2Value;

  encoder1Value = getEncoderValue(1);
  Serial.print("Encoder X= ");
  Serial.println(encoder1Value);

  encoder2Value = getEncoderValue(2);
  Serial.print(" Encoder Y= ");
  Serial.println(encoder2Value);

  if (oldX != encoder1Value || oldY != encoder2Value) {
    Serial1.print(encoder1Value);
    Serial1.print("  / ");
    Serial1.println(encoder2Value);
    oldX = encoder1Value;
    oldY = encoder2Value;
  }
}

void PS2_key() {
//  ps2x.read_gamepad(false, vibrate); //read controller and set large motor to spin at 'vibrate' speed
  if (ps2x.Button(PSB_START))        //will be TRUE as long as button is pressed
    Serial.println("Start is being held");
  if (ps2x.Button(PSB_SELECT))
    Serial.println("Select is being held");

  if (ps2x.Button(PSB_PAD_UP)) {     //will be TRUE as long as button is pressed
    Serial.print("Up held this hard: ");
    Serial.println(ps2x.Analog(PSAB_PAD_UP), DEC);
  }
  if (ps2x.Button(PSB_PAD_RIGHT)) {
    Serial.print("Right held this hard: ");
    Serial.println(ps2x.Analog(PSAB_PAD_RIGHT), DEC);
  }
  if (ps2x.Button(PSB_PAD_LEFT)) {
    Serial.print("LEFT held this hard: ");
    Serial.println(ps2x.Analog(PSAB_PAD_LEFT), DEC);
  }
  if (ps2x.Button(PSB_PAD_DOWN)) {
    Serial.print("DOWN held this hard: ");
    Serial.println(ps2x.Analog(PSAB_PAD_DOWN), DEC);
  }

  vibrate = ps2x.Analog(PSAB_CROSS);  //this will set the large motor vibrate speed based on how hard you press the blue (X) button
  if (ps2x.NewButtonState()) {        //will be TRUE if any button changes state (on to off, or off to on)
    if (ps2x.Button(PSB_L3))
      Serial.println("L3 pressed");
    if (ps2x.Button(PSB_R3))
      Serial.println("R3 pressed");
    if (ps2x.Button(PSB_L2))
      Serial.println("L2 pressed");
    if (ps2x.Button(PSB_R2))
      Serial.println("R2 pressed");
    if (ps2x.Button(PSB_TRIANGLE))
      Serial.println("Triangle pressed");
  }

  if (ps2x.ButtonPressed(PSB_CIRCLE))              //will be TRUE if button was JUST pressed
    Serial.println("Circle just pressed");
  if (ps2x.NewButtonState(PSB_CROSS))              //will be TRUE if button was JUST pressed OR released
    Serial.println("X just changed");
  if (ps2x.ButtonReleased(PSB_SQUARE))             //will be TRUE if button was JUST released
    Serial.println("Square just released");

  if (ps2x.Button(PSB_L1) || ps2x.Button(PSB_R1)) { //print stick values if either is TRUE
    Serial.print("Stick Values:");
    Serial.print(ps2x.Analog(PSS_LY), DEC); //Left stick, Y axis. Other options: LX, RY, RX
    Serial.print(",");
    Serial.print(ps2x.Analog(PSS_LX), DEC);
    Serial.print(",");
    Serial.print(ps2x.Analog(PSS_RY), DEC);
    Serial.print(",");
    Serial.println(ps2x.Analog(PSS_RX), DEC);
  }
}

void steer_control(int st) {
  bool LR;
  st > 0 ? LR = 1 : LR = 0;
//  st > 3 ? digitalWrite(LL, HIGH) : digitalWrite(LL, LOW);
//  st < -3 ? digitalWrite(RL, HIGH) : digitalWrite(RL, LOW);
  abs(st) < 2 ? st = 0 : st ;
  digitalWrite(MOT3F, LR);
  digitalWrite(MOT3R, !LR);
  analogWrite(MOT3E, abs(st * 12));
}