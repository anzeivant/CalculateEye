#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  140  // 最小脉宽
#define SERVOMAX  520  // 最大脉宽

// 舵机脉宽值的默认设置
int lexpulse = 300;  // 左眼脉宽
int rexpulse = 300;  // 右眼脉宽
int leypulse = 300;  // 左眼上下脉宽
int reypulse = 300;  // 右眼上下脉宽

void setup() {
  Serial.begin(9600);  // 启动串口通信
  pwm.begin();
  pwm.setPWMFreq(60);  // 设置舵机的更新频率
  delay(10);
}

// 处理眼球滚动
void handle_roll_eye(int direction) {
  // 根据 direction 控制眼球的左右或上下运动
  switch (direction) {
    case 0:
      // 眼球旋转一圈（需要实际的舵机动作）
      lexpulse = 320;
      rexpulse = 320;
      break;
    case 1:
      // 眼球向左下方转动
      lexpulse = 220;
      rexpulse = 220;
      leypulse = 500;
      reypulse = 500;
      break;
    case 2:
      // 眼球向下方转动
      lexpulse = 250;
      rexpulse = 250;
      leypulse = 600;
      reypulse = 600;
      break;
    case 3:
      // 眼球向右下方转动
      lexpulse = 380;
      rexpulse = 380;
      leypulse = 500;
      reypulse = 500;
      break;
    case 4:
      // 眼球向左方转动
      lexpulse = 220;
      rexpulse = 220;
      leypulse = 300;
      reypulse = 300;
      break;
    case 5:
      // 眼球静静地看着你
      lexpulse = 300;
      rexpulse = 300;
      leypulse = 300;
      reypulse = 300;
      break;
    case 6:
      // 眼球向右方转动
      lexpulse = 380;
      rexpulse = 380;
      leypulse = 300;
      reypulse = 300;
      break;
    case 7:
      // 眼球向左上方转动
      lexpulse = 220;
      rexpulse = 220;
      leypulse = 200;
      reypulse = 200;
      break;
    case 8:
      // 眼球向上方转动
      lexpulse = 250;
      rexpulse = 250;
      leypulse = 200;
      reypulse = 200;
      break;
    case 9:
      // 眼球向右上方转动
      lexpulse = 380;
      rexpulse = 380;
      leypulse = 200;
      reypulse = 200;
      break;
    default:
      Serial.println("未知方向");
      return;
  }

  // 控制舵机位置
  pwm.setPWM(0, 0, lexpulse);  // 左右
  pwm.setPWM(1, 0, leypulse);  // 上下

  // 打印方向信息
  Serial.print("眼球转动方向: ");
  Serial.println(direction);
}

// 处理眼球眨眼
void handle_twink_eye(int count) {
  // 根据 count 控制眼睛眨眼次数
  for (int i = 0; i < count; i++) {
    pwm.setPWM(2, 0, 400);
    pwm.setPWM(3, 0, 240);
    pwm.setPWM(4, 0, 240);
    pwm.setPWM(5, 0, 400);
    delay(100);  // 延时眨眼动作
    pwm.setPWM(2, 0, 520);  // 恢复正常位置
    pwm.setPWM(3, 0, 520);
    pwm.setPWM(4, 0, 520);
    pwm.setPWM(5, 0, 520);
    delay(100);  // 延时
  }
  Serial.print("TWINK count: ");
  Serial.println(count);  // 打印眨眼次数
}


void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // 读取一行数据
    input.trim();  // 去除首尾空格和换行符

    if (input.startsWith("ROLL:")) {
      int direction = input.substring(5).toInt();  // 获取方向值
      handle_roll_eye(direction);  // 执行眼球转动操作
    }
    else if (input.startsWith("TWINK:")) {
      int count = input.substring(6).toInt();  // 获取眨眼次数
      handle_twink_eye(count);  // 执行眨眼操作
    }
  }

  delay(5);
}

