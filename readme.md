# 🎵 智能灵巧手 - 手势识别音乐控制系统

<div align="center">


**一个集成手势识别、音乐演奏和3D打印的开源灵巧手控制系统**

[English](README_EN.md) | 简体中文

</div>

---

## 📖 项目简介

这是一个完整的开源灵巧手控制系统，通过计算机视觉实时识别用户手势，驱动机械灵巧手做出相应动作。系统创新性地将手指弯曲动作映射为音乐音符，实现了类似"音乐古筝"的交互体验，并内置了《沧海一声笑》音乐游戏模式。

### ✨ 核心特性

- 🤖 **实时手势识别** - 基于摄像头的高精度手势跟踪与识别
- 🎹 **音乐演奏模式** - 手指弯曲控制音符，打造独特的音乐体验
- 🎮 **互动游戏模式** - 内置《沧海一声笑》节奏游戏
- 🦾 **灵巧手控制** - 精确的多自由度手指运动控制
- 🖨️ **完整3D模型** - 可直接打印的灵巧手结构设计
- 📦 **全套硬件方案** - 包含BOM表、电路图和组装教程
- 🔧 **开箱即用** - 详细的文档和教程，易于复现

---

## 🎬 演示视频

> 📹 在这里添加项目演示视频或 GIF

---

## 🗂️ 项目结构

```
├── software/                  # 软件代码
│   ├── gesture_recognition/   # 手势识别模块
│   ├── hand_control/          # 灵巧手控制模块
│   ├── music_engine/          # 音乐播放引擎
│   └── game_mode/             # 游戏模式实现
├── hardware/                  # 硬件资料
│   ├── 3d_models/            # 3D打印模型 (STL/STEP)
│   ├── circuits/             # 电路原理图和PCB设计
│   ├── bom/                  # 物料清单 (BOM)
│   └── assembly/             # 组装说明文档
├── docs/                      # 项目文档
│   ├── user_manual.md        # 用户手册
│   ├── api_reference.md      # API参考文档
│   └── troubleshooting.md    # 故障排除指南
├── assets/                    # 资源文件
│   └── music/                # 音乐文件和音效
├── examples/                  # 示例代码
├── requirements.txt           # Python依赖
└── README.md                 # 项目说明
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8 
- OpenCV 4.5+
- 支持的操作系统：Windows 10/11
- 摄像头（推荐720p或更高分辨率）
- 灵巧手硬件设备

### 软件安装

1. **克隆仓库**
```bash
git clone https://github.com/yourusername/dexterous-hand-control.git
cd dexterous-hand-control
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **运行程序**
```bash
# 启动手势识别系统
python main.py

# 仅启动音乐模式
python main.py --mode music

# 启动游戏模式
python main.py --mode game
```

### 硬件组装

详细的硬件组装步骤请参考：[组装教程](hardware/assembly/README.md)

**快速概览：**
1. 3D打印所有零件（文件位于 `hardware/3d_models/`）
2. 根据BOM表采购电子元件（`hardware/bom/`）
3. 按照电路图焊接电路板（`hardware/circuits/`）
4. 按照组装说明进行机械装配
5. 烧录固件并连接到电脑

---

## 🎮 功能模块

### 1. 手势识别模式

使用MediaPipe或其他计算机视觉技术，实时追踪手部21个关键点，识别手势动作并控制灵巧手模仿。

### 2. 音乐古筝模式

将手指的弯曲程度映射为音乐音符，创造独特的音乐演奏体验。

**特性：**
- 五指对应不同音符（可自定义音阶）
- 实时音频反馈
- 支持录制和回放
- 可调节音色和音量

### 3. 游戏模式 - 《沧海一声笑》

节奏游戏模式，跟随音乐节拍做出相应手势。

**游戏玩法：**
- 音符从屏幕上方下落
- 按照提示做出对应手势
- 根据准确度和时机评分
- 支持难度选择

---

## 📚 文档

- [用户手册](docs/user_manual.md) - 详细的使用说明
- [API参考](docs/api_reference.md) - 开发者接口文档
- [硬件组装指南](hardware/assembly/README.md) - 完整的组装步骤
- [故障排除](docs/troubleshooting.md) - 常见问题解决方案
- [贡献指南](CONTRIBUTING.md) - 如何为项目做贡献

---

## 🛠️ 技术栈

### 软件
- **Python** - 主要编程语言
- **OpenCV** - 计算机视觉处理
- **MediaPipe** - 手势识别
- **PyAudio / pygame** - 音频处理
- **NumPy / SciPy** - 数值计算
- **Serial Communication** - 硬件通信

### 硬件
- **Arduino / ESP32** - 微控制器
- **舵机** - 执行器
- **传感器** - 位置反馈
- **3D打印材料** - PLA / PETG

---

## 🎯 路线图

- [x] 基础手势识别
- [x] 灵巧手控制系统
- [x] 音乐模式实现
- [x] 游戏模式开发
- [x] 3D模型设计
- [ ] 移动端App开发
- [ ] 机器学习手势自定义
- [ ] 多人联机模式
- [ ] 更多音乐游戏曲目

---

## 🤝 贡献

我们欢迎所有形式的贡献！无论是新功能、bug修复、文档改进还是问题反馈。

请参阅 [贡献指南](CONTRIBUTING.md) 了解详细信息。

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 👥 致谢

感谢所有为这个项目做出贡献的开发者和用户！

特别感谢：
- OpenCV 和 MediaPipe 团队
- 3D打印社区的支持
- 所有提供反馈和建议的用户

---

## 📧 联系方式

- **项目主页**: [https://github.com/yourusername/dexterous-hand-control](https://github.com/HBC-Tech-coder/RobotHand)
- **问题反馈**: [[Issues](https://github.com/yourusername/dexterous-hand-control/issues)](https://github.com/HBC-Tech-coder/RobotHand/issues)
- **邮箱**: 
- **讨论区**: [Discussions](https://github.com/yourusername/dexterous-hand-control/discussions)

---

## ⭐ Star History

如果这个项目对你有帮助，请给我们一个 Star ⭐️

[![Star History Chart](https://api.star-history.com/svg?repos=HBC-Tech-coder/RobotHand&type=Date)](https://star-history.com/#HBC-Tech-coder/RobotHand&Date)

---

<div align="center">

**Made with ❤️ HBC Tech**

</div>
