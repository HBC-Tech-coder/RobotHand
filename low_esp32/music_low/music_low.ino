#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// 手指伸直/弯曲的PWM配置参数
// #define wrist_straighten       102
// #define wrist_flex            502
// #define indexFinger_straighten 550
// #define indexFinger_flex       102
// #define middle_straighten      600  
// #define middle_flex            208
// #define ring_straighten        102
// #define ring_flex              490
// #define thumb_straighten       550
// #define thumb_flex             312
// #define pinky_straighten       102
// #define pinky_flex             480

// music_1
// #define wrist_straighten       102
// #define wrist_flex            502
// #define indexFinger_straighten 120
// #define indexFinger_flex       380
// #define middle_straighten      470  
// #define middle_flex            150
// #define ring_straighten        450
// #define ring_flex              150
// #define thumb_straighten       120
// #define thumb_flex             280
// #define pinky_straighten       500
// #define pinky_flex             200

// music_2
#define wrist_straighten       102
#define wrist_flex            502
#define indexFinger_straighten 120
#define indexFinger_flex       380
#define middle_straighten      450  
#define middle_flex            180
#define ring_straighten        500
#define ring_flex              250
#define thumb_straighten       110
#define thumb_flex             270
#define pinky_straighten       500
#define pinky_flex             250
#define SERVO_FREQ 50
#define MAX_ITERATIONS 150
#define STEP_SIZE 10
#define GESTURE_LENGTH 6

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

bool state0[GESTURE_LENGTH] = {false, false, false, false, false, false};
bool state1[GESTURE_LENGTH] = {false, false, false, false, false, false};
bool change = false;
char sData;
String state;

// 手指对应的舵机通道
const int wrist = 0;
const int indexFinger = 1;
const int middle = 2;
const int ring = 3;
const int thumb = 4;
const int pinky = 5;

const int fingerPins[] = {wrist, indexFinger, middle, ring, thumb, pinky};

// 获取指定手指的伸直/弯曲PWM值
void getPwmRange(int fingerId, int &straighten, int &flex) {
  switch(fingerId) {
    case wrist:
      straighten = wrist_straighten;
      flex = wrist_flex;
      break;
    case indexFinger:
      straighten = indexFinger_straighten;
      flex = indexFinger_flex;
      break;
    case middle:
      straighten = middle_straighten;
      flex = middle_flex;
      break;
    case ring:
      straighten = ring_straighten;
      flex = ring_flex;
      break;
    case thumb:
      straighten = thumb_straighten;
      flex = thumb_flex;
      break;
    case pinky:
      straighten = pinky_straighten;
      flex = pinky_flex;
      break;
    default:
      straighten = 102;
      flex = 502;
  }
}

// 初始化舵机到伸直位置
void initializeServos() {
  Serial.println("Initializing servos...");
  for (int i = 0; i < GESTURE_LENGTH; i++) {
    int straighten, flex;
    getPwmRange(i, straighten, flex);
    pwm.setPWM(fingerPins[i], 0, straighten);
  }
  delay(1000);
}

// 验证手势数据是否有效
bool validateGestureData(String data) {
  if (data.length() != GESTURE_LENGTH) {
    Serial.println("Error: Invalid data length");
    return false;
  }
  for (int i = 0; i < GESTURE_LENGTH; i++) {
    char c = data.charAt(i);
    if (c != '0' && c != '1') {
      Serial.println("Error: Invalid character in gesture data");
      return false;
    }
  }
  return true;
}

// 检查手指状态是否变化
bool hasStateChanged() {
  for (int i = 0; i < GESTURE_LENGTH; i++) {
    if (state0[i] != state1[i]) {
      return true;
    }
  }
  return false;
}

// 数据接收任务函数
void receiveDataCode(void * parameter) {
  for (;;) {
    while (Serial.available()) {
      sData = Serial.read();
      delay(2);
      if (sData == '\n') {
        if (validateGestureData(state)) {
          for (int i = 0; i < GESTURE_LENGTH; i++) {
            state0[i] = (state.charAt(i) == '1');
          }
          change = true;
          Serial.print("Received: ");
          Serial.println(state);
        }
        state = "";
      } else if (sData >= '0' && sData <= '1') {
        state += sData;
      }
      if (state.length() > GESTURE_LENGTH) {
        state = "";
      }
    }
    delay(2);
  }
}

// 移动手指函数
void moveFinger(int fingerId, bool targetFlex, int iteration) {
  int straighten, flex;
  getPwmRange(fingerId, straighten, flex);
  int startPwm = targetFlex ? straighten : flex;
  int endPwm = targetFlex ? flex : straighten;
  float progress = (float)iteration / MAX_ITERATIONS;
  int currentPwm = startPwm + (endPwm - startPwm) * progress;
  pwm.setPWM(fingerId, 0, currentPwm);
}

TaskHandle_t receiveData; // 任务句柄

void setup() {
  Serial.begin(9600);
  Serial.println("ESP32 Hand Control Started");

  Wire.begin();
  pwm.begin();
  pwm.setOscillatorFrequency(25000000);
  pwm.setPWMFreq(SERVO_FREQ);

  initializeServos();

  // 创建数据接收任务
  xTaskCreatePinnedToCore(
    receiveDataCode,
    "receiveData",
    10000,
    NULL,
    1,
    &receiveData,
    0);
}

void loop() {
  if (change && hasStateChanged()) {
    Serial.println("Processing gesture change...");
    for (int i = 0; i <= MAX_ITERATIONS; i += STEP_SIZE) {
      for (int j = 0; j < GESTURE_LENGTH; j++) {
        if (state0[j] != state1[j]) {
          moveFinger(fingerPins[j], state0[j], i);
        }
      }
      delay(5);
    }
    memcpy(state1, state0, sizeof(state0));
    change = false;
    Serial.print("Current state: ");
    for (bool s : state1) Serial.print(s ? "1" : "0");
    Serial.println();
  }
  delay(5);
}