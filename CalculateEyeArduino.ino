#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  140
#define SERVOMAX  520

int xval = 512;  // 初始值居中
int yval = 512;  // 初始值居中
int trimval = 0;
int switchval = 0;

int lexpulse;
int rexpulse;
int leypulse;
int reypulse;
int uplidpulse;
int lolidpulse;
int altuplidpulse;
int altlolidpulse;

void setup() {
  Serial.begin(9600);
  Serial.println("Serial control for eye mechanism");
  
  pwm.begin();
  pwm.setPWMFreq(60);
  delay(10);
}

void loop() {
  // 检查是否有串口输入
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // 读取一行数据
    input.trim();  // 去掉可能的空格和换行符

    // 解析输入命令
    if (input.startsWith("X:")) {
      xval = input.substring(2).toInt();  // 提取数值
      xval = constrain(xval, 0, 1023);   // 限制范围
    } else if (input.startsWith("Y:")) {
      yval = input.substring(2).toInt();
      yval = constrain(yval, 0, 1023);
    } else if (input.startsWith("T:")) {
      trimval = input.substring(2).toInt();
      trimval = constrain(trimval, -40, 40);
    } else if (input.startsWith("B:")) {
      switchval = input.substring(2).toInt();
      switchval = constrain(switchval, 0, 1);
    }
  }

  // 计算舵机脉宽值
  lexpulse = map(xval, 0, 1023, 220, 440);
  rexpulse = lexpulse;

  leypulse = map(yval, 0, 1023, 250, 500);
  reypulse = map(yval, 0, 1023, 400, 280);

  uplidpulse = map(yval, 0, 1023, 400, 280);
  uplidpulse -= (trimval - 40);
  uplidpulse = constrain(uplidpulse, 280, 400);
  altuplidpulse = 680 - uplidpulse;

  lolidpulse = map(yval, 0, 1023, 410, 280);
  lolidpulse += (trimval / 2);
  lolidpulse = constrain(lolidpulse, 280, 400);
  altlolidpulse = 680 - lolidpulse;

  // 根据 switchval 控制眼皮位置
  if (switchval == 1) {
    pwm.setPWM(2, 0, 400);
    pwm.setPWM(3, 0, 240);
    pwm.setPWM(4, 0, 240);
    pwm.setPWM(5, 0, 400);
  } else {
    pwm.setPWM(2, 0, uplidpulse);
    pwm.setPWM(3, 0, lolidpulse);
    pwm.setPWM(4, 0, altuplidpulse);
    pwm.setPWM(5, 0, altlolidpulse);
  }

  // 控制眼球位置
  pwm.setPWM(0, 0, lexpulse);
  pwm.setPWM(1, 0, leypulse);

  // 打印调试信息
  Serial.print("X:"); Serial.print(xval);
  Serial.print(" Y:"); Serial.print(yval);
  Serial.print(" T:"); Serial.print(trimval);
  Serial.print(" B:"); Serial.println(switchval);

  delay(5);  // 延时以稳定舵机动作
}
