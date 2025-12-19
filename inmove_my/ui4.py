import cv2
import mediapipe as mp
import time
import serial
import threading
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QFrame, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from test7 import get_frame_generator
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import traceback
import pygame
from pygame import mixer

class HandDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.handedness = None  # 存储手的左右信息

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            self.handedness = []
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                if draw:
                    self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                # 获取手的左右信息
                self.handedness.append(handedness.classification[0].label)
        return frame
    
    def findPosition(self, frame, handNo=0, draw=False):
        lmList = []
        handType = None

        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                if self.handedness and handNo < len(self.handedness):
                    handType = self.handedness[handNo]

                for id, lm in enumerate(myHand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    lmList.append([id, cx, cy])

                    if draw and id == 0:
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
        return lmList, handType

def serial_monitor(ser, status_signal):
    """独立线程监听Arduino串口输出"""
    while True:
        try:
            if ser.in_waiting > 0:
                arduino_data = ser.readline().decode('utf-8').strip()
                if arduino_data:
                    status_signal.emit(f"[Arduino]: {arduino_data}")
        except Exception as e:
            status_signal.emit(f"串口连接异常: {str(e)}")
            break
        time.sleep(0.01)

def draw_text_with_chinese(frame, text, position, font_size=16, color=(255, 255, 0)):
    """使用PIL绘制中文文本（适配小屏幕字体）"""
    try:
        # 将OpenCV的BGR格式转为PIL的RGB格式
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 设置中文字体路径
        font_path = None
        try:
            # Windows系统默认中文字体
            font_path = "C:/Windows/Fonts/simhei.ttf"
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        except:
            try:
                # Linux系统默认中文字体
                font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            except:
                try:
                    # macOS系统默认中文字体
                    font_path = "/System/Library/Fonts/PingFang.ttc"
                    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                except:
                    # 如果都找不到，使用默认字体
                    font = ImageFont.load_default()
                    print("警告: 未找到中文字体，使用默认字体")
        
        # 绘制文本
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))  # PIL使用RGB顺序
        
        # 将PIL图像转回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"文本绘制错误: {e}")
        # 出错时返回原始帧
        return frame

class VideoThread(QThread):
    """视频处理线程"""
    update_frame = pyqtSignal(np.ndarray)
    update_status = pyqtSignal(str)
    
    def __init__(self, detector, ser, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.ser = ser
        self._main_window = parent  # 存储父窗口引用
        self.running = False
        self.prev_finger_state = "000000"  # 初始手指状态
        self.finger_changed = False
        self.demo_mode = False
        self.demo_patterns = ["000000","001111","000111","000011","000010","000000","011111","000000"]
        self.demo_index = 0
        self.demo_timer = 0
        self.demo_interval = 1.5  # 秒
        self.hand = [["手腕", False], ["食指", False], ["中指", False], 
                    ["无名指", False], ["拇指", False], ["小指", False]]
        self.frame_count = 0
        self.PROCESSING_INTERVAL = 1
        self.WINDOW_SIZE = 2
        self.finger_history = {
            0: deque(maxlen=self.WINDOW_SIZE),  # 手腕
            1: deque(maxlen=self.WINDOW_SIZE),  # 食指
            2: deque(maxlen=self.WINDOW_SIZE),  # 中指
            3: deque(maxlen=self.WINDOW_SIZE),  # 无名指
            4: deque(maxlen=self.WINDOW_SIZE),  # 拇指
            5: deque(maxlen=self.WINDOW_SIZE)   # 小指
        }
        for i in range(self.WINDOW_SIZE):
            for finger in self.finger_history:
                self.finger_history[finger].append(False)
        
        # 视频优化参数
        self.resize_frame = True  # 是否调整帧尺寸
        self.target_width = 640   # 目标宽度（小屏幕优化）
        self.target_height = 480  # 目标高度
        self.skip_frames = 1      # 跳帧处理，每N帧处理1帧
        self.current_skip = 0
        
    def run(self):
        try:
            self.running = True
            prevTime = 0
            
            # 打开摄像头（添加错误处理）
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.update_status.emit("摄像头打开失败")
                return
                
            # 设置摄像头分辨率（小屏幕优化）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.update_status.emit(f"系统就绪，正在检测手势...")

            while self.running:
                # 演示模式处理
                if self.demo_mode:
                    current_time = time.time()
                    if current_time - self.demo_timer > self.demo_interval:
                        self.demo_timer = current_time
                        pattern = self.demo_patterns[self.demo_index]
                        self.send_finger_status(pattern)
                        self.demo_index = (self.demo_index + 1) % len(self.demo_patterns)
                        continue  # 跳过正常检测流程
                
                ret, frame = cap.read()
                if not ret:
                    self.update_status.emit("读取帧失败")
                    break
                    
                self.frame_count += 1
                self.current_skip += 1
                
                # 跳帧处理，减少计算量
                if self.current_skip <= self.skip_frames:
                    continue
                self.current_skip = 0
                
                # 调整帧尺寸（如果原始尺寸过大）
                if self.resize_frame and (frame.shape[1] > self.target_width or frame.shape[0] > self.target_height):
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
                
                # 水平镜像画面（保持检测逻辑不变）
                frame = cv2.flip(frame, 1)
                
                # 始终检测手部并绘制关键点
                frame = self.detector.findHands(frame)
                lmList, handType = self.detector.findPosition(frame)
                
                # 每帧都检测手指状态，但只在必要时更新平均值
                current_state = [False] * 6  # 初始化当前帧的手指状态
                
                if len(lmList) > 0: 
                    j = 1
                    
                    for i in range(1, 6):
                            if i == 1:  # 拇指检测
                                # 根据左右手决定是否取反
                                if (handType == "Left" and lmList[4][1] <= lmList[3][1]) or \
                                   (handType == "Right" and lmList[4][1] > lmList[3][1]):
                                    current_state[4] = True  # 拇指弯曲
                            else:  # 其他四指检测
                                finger_tip = i * 4
                                finger_pip = i * 4 - 2
                                
                                if finger_tip < len(lmList) and finger_pip < len(lmList):
                                    if lmList[finger_tip][2] > lmList[finger_pip][2]:
                                        current_state[j] = True  # 手指弯曲
                                
                                if j == 3:
                                    j += 2
                                else:
                                    j += 1
                
                # 更新滑动窗口数据
                for i in range(6):
                    self.finger_history[i].append(current_state[i])
                
                # 每5帧计算一次平均值并决定最终状态
                if self.frame_count % self.WINDOW_SIZE == 0:
                    change = False
                    threshold = 1  # 窗口大小为2时，只需1帧为True则认为弯曲
                    
                    for i in range(6):
                        # 计算平均值
                        count_true = sum(self.finger_history[i])
                        new_state = count_true > threshold
                        
                        # 如果状态变化，记录变化
                        if new_state != self.hand[i][1]:
                            self.hand[i][1] = new_state
                            change = True
                            self.update_status.emit(f"[Python] Frame {self.frame_count}: {self.hand[i][0]}: {'弯曲' if new_state else '伸直'}")
                    
                    # 如果状态变化，发送新命令
                    if change and self.ser and self.ser.is_open:
                        msg = ""
                        for i in range(6):
                            if self.hand[i][1]:
                                msg += "1"
                            else:
                                msg += "0"

                        msg = msg.strip()
                        # 检测手指状态变化
                        current_state = msg
                        self.finger_changed = current_state != self.prev_finger_state
                        self.prev_finger_state = current_state

                        print(f"finger stage: {current_state}, change: {self.finger_changed}")

                        if self.parent() is not None:
                            print(f"Parent exists - play_mode: {getattr(self.parent(), 'play_mode', False)}")
                        else:
                            print("Warning: Parent is None!")

                        # 如果手指状态变化且处于演奏模式，发送信号
                        # if self.finger_changed and hasattr(self.parent(), 'play_mode') and self.parent().play_mode:
                        if self.finger_changed and hasattr(self, '_main_window'):
                            print(f"finger stage: {current_state}")
                            self._main_window.last_boost_time = time.time()
                            self._main_window.set_volume(self._main_window.boost_volume)
                            self.update_status.emit(f"[音量提升] 检测到手势变化，音量提升至{int(self._main_window.boost_volume*100)}%")
                            # 仅提升音量，不发送信号给Arduino
                        
                        self.update_status.emit(f"[Python] Sending: {msg}")
                        self.send_finger_status(msg)
                
                # 计算并显示实际FPS（字体大小调整为18）
                currentTime = time.time()
                if prevTime != 0:
                    fps = 1 / (currentTime - prevTime)
                    frame = draw_text_with_chinese(frame, f"实际FPS: {int(fps)}", (10, 50), 18, (255, 0, 255))
                prevTime = currentTime
                
                # 显示处理参数（字体大小调整为18）
                frame = draw_text_with_chinese(frame, f"滑动窗口: {self.WINDOW_SIZE}帧", (10, 80), 18, (255, 255, 0))
                frame = draw_text_with_chinese(frame, f"帧计数: {self.frame_count}", (10, 110), 18, (255, 255, 0))

                # 添加状态显示（字体大小调整为16，间距缩小）
                y_offset = 140
                # 显示手的左右信息
                if handType:
                    frame = draw_text_with_chinese(
                        frame,
                        f"检测到: {handType}",
                        (10, y_offset),
                        16,
                        (255, 255, 255))
                    y_offset += 30
                
                for i, (name, state) in enumerate(self.hand):
                    color = (0, 255, 0) if state else (0, 0, 255)
                    frame = draw_text_with_chinese(
                        frame, 
                        f"{name}: {'弯曲' if state else '伸直'}", 
                        (10, y_offset + i * 30),  # 行间距缩小
                        16,  # 字体大小减小
                        color
                    )

                # 转换BGR到RGB用于Qt显示
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_frame.emit(frame)

            cap.release()
            self.update_status.emit("视频线程已停止")
        except Exception as e:
            self.update_status.emit(f"视频线程异常: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def stop(self):
        self.running = False
        self.wait()  # 等待线程安全退出

    def send_finger_status(self, finger_status):
        """
        发送手指状态到下位机
        :param finger_status: 6位字符串，如"011111"
        :return: bool 发送是否成功
        """
        if not self.ser or not self.ser.is_open:
            self.update_status.emit("串口未连接，无法发送")
            return False
        
        try:
            msg = finger_status + '\n'
            self.ser.write(msg.encode("ascii"))
            self.ser.flush()
            self.update_status.emit(f"[发送成功]: {msg.strip()}")
            return True
        except serial.SerialException as e:
            self.update_status.emit(f"串口发送失败: {str(e)}")
            return False
        except Exception as e:
            self.update_status.emit(f"发送异常: {str(e)}")
            return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化音频控制属性
        pygame.mixer.init()
        self.current_sound = None
        self.default_volume = 0.05  # 默认音量5%
        self.boost_volume = 0.9    # 手指变化时提升到的音量
        self.boost_duration = 0.5  # 音量提升持续时间(秒)
        self.last_boost_time = 0   # 上次音量提升时间
        
        # 图片显示相关
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.hide()
        
       # 设置窗口标题和初始大小
        self.setWindowTitle("手势控制系统")
        
        # 先初始化状态变量
        self.ser = None
        self.serial_thread = None
        self.is_running = False  # 移到这里，在init_ui之前初始化
        self.play_mode = False  # 演奏模式状态
        
        # 初始化UI
        self.init_ui()
        
        # 启动时全屏显示
        self.showFullScreen()
        
    def init_ui(self):
        # 获取可用串口列表
        self.available_ports = []
        try:
            import serial.tools.list_ports
            self.available_ports = [port.device for port in serial.tools.list_ports.comports()]
        except:
            self.available_ports = []
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 水平分割视频区和控制区
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 主视频区
        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.NoFrame)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # 手势视频显示 (主窗口)
        self.video_label = QLabel("等待视频流...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        video_layout.addWidget(self.video_label)
        
        # 添加图片显示层（作为窗口的直接子部件）
        self.image_label.setStyleSheet("background-color: transparent;")
        self.image_label.setParent(self)
        self.image_label.raise_()
        
        main_layout.addWidget(video_frame, 7)  # 主视频区占70%
        
        # 控制区 - 使用背景图片
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.NoFrame)
        control_frame.setStyleSheet("""
            QFrame {
                background-image: url('ui_background.png');
                background-repeat: no-repeat;
                background-position: center;
                background-size: cover;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(50, 70, 50, 30)
        control_layout.setSpacing(10)
        
        # 串口选择容器 - 用于控制位置
        port_container = QWidget()
        port_container.setStyleSheet("background-color: transparent;")
        port_container_layout = QHBoxLayout(port_container)
        port_container_layout.setContentsMargins(260, 126, 0, 0)  # 左边距80px，上边距20px
        
        # 串口选择下拉框（缩短宽度）
        self.port_combo = QComboBox()
        self.port_combo.addItems(self.available_ports)
        self.port_combo.setFixedSize(400, 40)  # 固定宽度280px，高度40px
        self.port_combo.setStyleSheet("""
            QComboBox {
                font-size: 10pt; 
                padding: 6px 12px;
                border: 2px solid #B8C5E0;
                border-radius: 20px;
                background-color: rgba(255, 255, 255, 0.95);
            }
            QComboBox:hover {
                border: 2px solid #7B9FD8;
                background-color: white;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #666;
                margin-right: 8px;
            }
        """)
        port_container_layout.addWidget(self.port_combo)
        port_container_layout.addStretch(1)  # 右侧添加弹性空间
        
        control_layout.addWidget(port_container)
        control_layout.addSpacing(120)
        
        # 按钮容器 - 居中对齐
        button_container = QWidget()
        button_container.setStyleSheet("background-color: transparent;")
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(12)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        # 主控制按钮 - 开始程序（加大尺寸）
        self.toggle_btn = QPushButton("开始程序")
        self.toggle_btn.setFixedSize(580, 56)
        self.update_button_style()
        self.toggle_btn.clicked.connect(self.toggle_program)
        button_layout.addWidget(self.toggle_btn, 0, Qt.AlignCenter)
        
        # 演示模式按钮（加大尺寸）
        self.demo_btn = QPushButton("启动演示模式")
        self.demo_btn.setFixedSize(580, 56)
        self.demo_btn.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                background-color: #6B8FD8;
                color: white;
                border-radius: 28px;
                border: none;
            }
            QPushButton:hover {
                background-color: #5A7DC7;
            }
            QPushButton:pressed {
                background-color: #4A6DB6;
            }
        """)
        self.demo_btn.clicked.connect(self.toggle_demo_mode)
        button_layout.addWidget(self.demo_btn, 0, Qt.AlignCenter)
        
        # 演奏模式按钮（加大尺寸）
        self.play_btn = QPushButton("开启演奏模式")
        self.play_btn.setFixedSize(580, 56)
        self.play_btn.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                background-color: #6B8FD8;
                color: white;
                border-radius: 28px;
                border: none;
            }
            QPushButton:hover {
                background-color: #5A7DC7;
            }
            QPushButton:pressed {
                background-color: #4A6DB6;
            }
        """)
        self.play_btn.clicked.connect(self.toggle_play_mode)
        button_layout.addWidget(self.play_btn, 0, Qt.AlignCenter)

        # 退出按钮（加大尺寸）
        self.exit_btn = QPushButton("退出程序")
        self.exit_btn.setFixedSize(580, 56)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                font-size: 11pt;
                background-color: rgba(255, 255, 255, 0.9);
                color: #D32F2F;
                border-radius: 28px;
                border: 2px solid #E0E0E0;
            }
            QPushButton:hover {
                background-color: white;
                border: 2px solid #D32F2F;
            }
            QPushButton:pressed {
                background-color: #F0F0F0;
            }
        """)
        self.exit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.exit_btn, 0, Qt.AlignCenter)
        
        control_layout.addWidget(button_container)
        
        control_layout.addSpacing(400)
        
        # 状态显示区容器 - 用于控制宽度
        status_container = QWidget()
        status_container.setStyleSheet("background-color: transparent;")
        status_container_layout = QHBoxLayout(status_container)
        status_container_layout.setContentsMargins(100, 0, 100, 0)

        # 状态显示区 - 透明文本框
        self.status_text = QLabel("系统未启动")
        self.status_text.setWordWrap(True)
        self.status_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status_text.setStyleSheet("""
            background-color: transparent;
            color: #333;
            font-size: 10pt;
            padding: 10px 15px;
            min-height: 60px;
        """)

        status_container_layout.addWidget(self.status_text)
        status_container_layout.addStretch(1)

        control_layout.addWidget(status_container)
        
        # 添加弹性空间
        control_layout.addStretch(1)
        
        # 音乐可视化显示区域
        viz_container = QWidget()
        viz_container.setStyleSheet("background-color: transparent;")
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(30, 0, 30, 20)  # 增加左右和底部边距

        self.music_viz_label = QLabel()
        self.music_viz_label.setFixedSize(500, 300)  # 从420x260调整为460x300
        self.music_viz_label.setStyleSheet("""
            background-color: rgba(40, 40, 40, 0.85); 
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
        """)
        self.music_viz_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(self.music_viz_label, 0, Qt.AlignCenter)
        control_layout.addWidget(viz_container)

        control_layout.addSpacing(20)  # 从10改为20
        
        main_layout.addWidget(control_frame, 3)  # 控制区占30%

    def update_button_style(self):
        """根据运行状态更新按钮样式"""
        if not self.is_running:
            # 开始状态 - 白色背景
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    font-size: 11pt;
                    background-color: rgba(255, 255, 255, 0.95);
                    color: #333;
                    border-radius: 28px;
                    border: 2px solid #B8C5E0;
                }
                QPushButton:hover {
                    background-color: white;
                    border: 2px solid #7B9FD8;
                }
                QPushButton:pressed {
                    background-color: #F0F0F0;
                }
            """)
        else:
            # 运行状态 - 蓝色背景
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    font-size: 11pt;
                    background-color: #6B8FD8;
                    color: white;
                    border-radius: 28px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #5A7DC7;
                }
                QPushButton:pressed {
                    background-color: #4A6DB6;
                }
            """)
            
    def toggle_program(self):
        """切换程序运行状态"""
        if not self.is_running:
            self.start_program()
        else:
            self.stop_program()

    def start_program(self):
        """开始程序按钮处理函数"""
        self.toggle_btn.setEnabled(False)  # 防止重复点击
        self.status_text.setText("系统正在启动...")
        
        try:
            # 打开串口（添加错误处理）
            selected_port = self.port_combo.currentText()
            if not selected_port and self.available_ports:
                selected_port = self.available_ports[0]
                
            self.ser = serial.Serial(
                port=selected_port,
                baudrate=9600,
                timeout=0.1,
                write_timeout=1
            )
            self.status_text.setText(f"串口 {self.ser.port} 打开成功")
            
            # 启动串口监听线程
            self.serial_thread = threading.Thread(
                target=serial_monitor, 
                args=(self.ser, self.update_status),
                daemon=True
            )
            self.serial_thread.start()
            
            # 启动视频处理线程
            self.detector = HandDetector(maxHands=1, detectionCon=0.7)
            self.video_thread = VideoThread(self.detector, self.ser, self)  # 传递self作为parent
            self.video_thread.update_frame.connect(self.update_video_frame)
            self.video_thread.update_status.connect(self.update_status)
            self.video_thread.start()
            
            # 更新状态
            self.is_running = True
            self.toggle_btn.setText("结束程序")
            self.update_button_style()
            self.toggle_btn.setEnabled(True)
            
        except serial.SerialException as e:
            self.status_text.setText(f"串口打开失败: {e}")
            self.toggle_btn.setEnabled(True)
        except Exception as e:
            self.status_text.setText(f"启动程序失败: {e}")
            self.toggle_btn.setEnabled(True)
    
    def toggle_demo_mode(self):
        """切换演示模式"""
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.demo_mode = not self.video_thread.demo_mode
            if self.video_thread.demo_mode:
                self.demo_btn.setText("停止演示模式")
                self.demo_btn.setStyleSheet("""
                    QPushButton {
                        font-size: 11pt;
                        background-color: #F44336;
                        color: white;
                        border-radius: 10px;
                        border: 2px solid #D32F2F;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                        border: 2px solid #C2185B;
                    }
                    QPushButton:pressed {
                        background-color: #C2185B;
                    }
                """)
                self.status_text.setText("演示模式已启动")
            else:
                self.demo_btn.setText("启动演示模式")
                self.demo_btn.setStyleSheet("""
                    QPushButton {
                        font-size: 11pt;
                        background-color: #9C27B0;
                        color: white;
                        border-radius: 10px;
                        border: 2px solid #7B1FA2;
                    }
                    QPushButton:hover {
                        background-color: #7B1FA2;
                        border: 2px solid #6A1B9A;
                    }
                    QPushButton:pressed {
                        background-color: #6A1B9A;
                    }
                """)
                self.status_text.setText("演示模式已停止")

    def set_volume(self, volume):
        """设置音量接口"""
        if hasattr(self, 'current_sound') and self.current_sound:
            self.current_sound.set_volume(volume)
            # 更新状态显示
            vol_percent = int(volume * 100)
            self.status_text.setText(f"音量设置为: {vol_percent}%")

    def check_volume_boost(self):
        """检查是否需要恢复默认音量"""
        current_time = time.time()
        if current_time - self.last_boost_time > self.boost_duration:
            self.set_volume(self.default_volume)
            # 添加状态更新
            current_text = self.status_text.text()
            if "音量恢复" not in current_text:
                self.status_text.setText(current_text + f"\n音量恢复默认: {int(self.default_volume*100)}%")

    def toggle_play_mode(self):
        """切换演奏模式"""
        self.play_mode = not self.play_mode
        if self.play_mode:
            self.play_btn.setText("关闭演奏模式")
            self.play_btn.setStyleSheet("""
                QPushButton {
                    font-size: 11pt;
                    background-color: #F44336;
                    color: white;
                    border-radius: 28px;
                    border: 2px solid #D32F2F;  Q
                }
                QPushButton:hover {
                    background-color: #D32F2F;
                    border: 2px solid #C2185B;
                }
                QPushButton:pressed {
                    background-color: #C2185B;
                }
            """)
            try:
                # 显示图片并居中
                screen = QApplication.desktop().screenGeometry()
                target_width = int(screen.width() * 2 / 3)
                
                # 加载图片并保持宽高比缩放
                pixmap = QPixmap("example.png")
                scaled_pixmap = pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
                
                # 计算居中位置
                x = (screen.width() - scaled_pixmap.width()) // 2
                y = (screen.height() - scaled_pixmap.height()) // 2
                
                # 设置图片位置和大小
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setGeometry(x, y, scaled_pixmap.width(), scaled_pixmap.height())
                self.image_label.show()
                
                # 3秒后开始播放音乐
                QTimer.singleShot(3000, self.start_playback)
            except Exception as e:
                self.status_text.setText(f"图片加载失败: {str(e)}")
                self.play_mode = False
                return
        else:
            self.play_btn.setText("开启演奏模式")
            self.play_btn.setStyleSheet("""
                QPushButton {
                    font-size: 11pt;
                    background-color: #6B8FD8;
                    color: white;
                    border-radius: 28px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #5A7DC7;
                }
                QPushButton:pressed {
                    background-color: #4A6DB6;
                }
            """)
            # 停止音频播放
            if hasattr(self, 'current_sound') and self.current_sound:
                self.current_sound.stop()
                self.current_sound = None
            
            # 停止音量检查定时器
            if hasattr(self, 'volume_timer'):
                self.volume_timer.stop()
                
            # 停止音乐可视化
            if hasattr(self, 'music_timer'):
                self.music_timer.stop()
            self.music_viz_label.hide()
            self.status_text.setText("演奏模式已关闭")

    def stop_program(self):
        """结束程序按钮处理函数"""
        self.toggle_btn.setEnabled(False)  # 防止重复点击
        self.status_text.setText("系统正在关闭...")
        
        # 停止音乐可视化
        if hasattr(self, 'music_timer'):
            self.music_timer.stop()
        self.music_viz_label.clear()
        
        # 确保演示模式也被关闭
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.demo_mode = False
        
        # 停止视频线程
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
        
        # 关闭串口
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.status_text.setText("串口已关闭")
        
        # 更新状态
        self.is_running = False
        self.toggle_btn.setText("开始程序")
        self.update_button_style()
        self.video_label.setText("等待视频流...")
        self.status_text.setText("系统已停止")
        self.toggle_btn.setEnabled(True)
    
    def start_playback(self):
        """3秒后开始播放音乐"""
        try:
            # 隐藏图片
            self.image_label.hide()
            
            # 加载并播放音频
            self.current_sound = mixer.Sound("audio/canhaiyi.wav")
            self.current_sound.set_volume(self.default_volume)
            self.current_sound.play(0)  # 0表示一次性播放

            # 启动音量检查定时器
            self.volume_timer = QTimer()
            self.volume_timer.timeout.connect(self.check_volume_boost)
            self.volume_timer.start(100)  # 每100ms检查一次
            
            # 启动音乐可视化
            self.music_generator = get_frame_generator()
            self.music_timer = QTimer()
            self.music_timer.timeout.connect(self.update_music_viz)
            self.music_timer.start(1000//60)  # 60 FPS
            self.music_viz_label.show()
            self.status_text.setText("演奏模式已启动 - 正在播放音频")
        except Exception as e:
            self.status_text.setText(f"音频播放失败: {str(e)}")
            self.play_mode = False

    def update_music_viz(self):
        """更新音乐可视化显示"""
        try:
            frame = next(self.music_generator)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.music_viz_label.setPixmap(QPixmap.fromImage(q_img))
        except StopIteration:
            self.music_timer.stop()
            self.music_viz_label.setText("音乐播放结束")

    def update_video_frame(self, frame):
        """更新视频帧显示，确保铺满视频区域"""
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # 让视频帧自适应视频标签大小，保持比例并平滑缩放
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def update_status(self, message):
        """更新状态文本"""
        self.status_text.setText(message)
    
    def resizeEvent(self, event):
        """窗口大小变化时，更新视频显示"""
        if hasattr(self, 'video_label') and self.video_label.pixmap():
            self.video_label.setPixmap(self.video_label.pixmap().scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        super().resizeEvent(event)
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.stop_program()
        event.accept()

if __name__ == "__main__":
    # 添加全局异常处理
    def exception_hook(exctype, value, tb):
        print(f"全局异常捕获: {exctype}, {value}")
        print("".join(traceback.format_exception(exctype, value, tb)))  # 修复这里
        sys._excepthook(exctype, value, tb)
        sys.exit(1)
    
    sys._excepthook = sys.excepthook
    sys.excepthook = exception_hook
    
    app = QApplication(sys.argv)
    # 设置全局字体，确保中文显示正常 
    font = app.font()
    font.setFamily("SimHei")  # Windows/Linux默认中文字体
    app.setFont(font)
    
    window = MainWindow()
    sys.exit(app.exec_())
