#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SERVO_FREQ 50  // PWM频率(Hz)
#define DEFAULT_PWM 300  // 默认PWM值
#define NUM_CHANNELS 16  // PCA9685的16个通道

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// 存储每个通道的当前PWM值
int channelValues[NUM_CHANNELS] = {0};

void setup() {
  Serial.begin(9600);
  Serial.println("PWM Controller Started");
  Serial.println("Usage: 'C[0-15]P[value]' to set channel PWM");
  Serial.println("Example: 'C0P300' sets channel 0 to PWM 300");
  
  Wire.begin();
  pwm.begin();
  pwm.setOscillatorFrequency(25000000);
  pwm.setPWMFreq(SERVO_FREQ);
  
  // 初始化所有通道为默认PWM值
  for (int i = 0; i < NUM_CHANNELS; i++) {
    channelValues[i] = DEFAULT_PWM;
    pwm.setPWM(i, 0, DEFAULT_PWM);
  }
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // 解析命令格式: C[channel]P[value]
    if (command.startsWith("C") && command.indexOf("P") > 0) {
      int channel = command.substring(1, command.indexOf("P")).toInt();
      int value = command.substring(command.indexOf("P") + 1).toInt();
      
      if (channel >= 0 && channel < NUM_CHANNELS && value >= 0 && value < 4096) {
        channelValues[channel] = value;
        pwm.setPWM(channel, 0, value);
        Serial.print("Set channel ");
        Serial.print(channel);
        Serial.print(" to PWM ");
        Serial.println(value);
      } else {
        Serial.println("Invalid channel or PWM value");
      }
    } else if (command.equals("READALL")) {
      // 读取所有通道的当前PWM值
      Serial.println("Current PWM values:");
      for (int i = 0; i < NUM_CHANNELS; i++) {
        Serial.print("Channel ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(channelValues[i]);
      }
    } else {
      Serial.println("Invalid command format. Use 'C[0-15]P[value]'");
    }
  }
  
  delay(10);
}