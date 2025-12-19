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
from PyQt5.QtWidgets import QListView
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
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 左侧视频区域（黑色背景）
        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: #1a1a1a;")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # 手势视频显示
        self.video_label = QLabel("等待视频流...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a; color: white; font-size: 16pt;")
        video_layout.addWidget(self.video_label)
        
        # 添加图片显示层
        self.image_label.setStyleSheet("background-color: transparent;")
        self.image_label.setParent(self)
        self.image_label.raise_()
        
        main_layout.addWidget(video_frame, 7)
        
        # 右侧控制区（使用背景图片）
        control_frame = QFrame()
        import os
        bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui_background2.png')
        
        if os.path.exists(bg_path):
            # 使用QLabel显示背景图片
            bg_label = QLabel(control_frame)
            from PyQt5.QtGui import QPixmap
            pixmap = QPixmap(bg_path)
            bg_label.setPixmap(pixmap)
            bg_label.setScaledContents(True)
            bg_label.lower()
            
            control_frame.setStyleSheet("background-color: transparent;")
        else:
            # 备用渐变背景
            control_frame.setStyleSheet("""
                QFrame {
                    background: qlineargradient(
                        x1:0, y1:0, x2:0, y2:1,
                        stop:0 #7B9FE8,
                        stop:1 #A4C2F4
                    );
                }
            """)
            print(f"警告: 未找到背景图片 {bg_path}")
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(100, 60, 100, 30)
        control_layout.setSpacing(0)
        
        # 跳过标题区域（图片中已有）
        control_layout.addSpacing(55)
        
        # COM串口选择 - 向右移动、变短、美化下拉菜单
        port_container = QWidget()
        port_container.setStyleSheet("background: transparent;")
        port_layout = QHBoxLayout(port_container)
        port_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧留白，使下拉框向右移动
        port_layout.addSpacing(150)  # 调整这个数字控制向右移动距离

        self.port_combo = QComboBox()
        self.port_combo.addItems(self.available_ports)
        self.port_combo.setFixedWidth(500)  # 固定宽度，使其变短

        # 设置下拉列表视图的样式（这是关键！）
        self.port_combo.setView(QListView())
        self.port_combo.view().window().setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.port_combo.view().window().setAttribute(Qt.WA_TranslucentBackground)

        self.port_combo.setStyleSheet("""
            QComboBox {
                font-size: 11pt;
                padding: 8px 20px;
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 20px;
                background-color: rgba(255, 255, 255, 0.3);
                color: white;
                min-height: 32px;
                font-weight: 500;
            }
            QComboBox:hover {
                background-color: rgba(255, 255, 255, 0.4);
                border: 2px solid rgba(255, 255, 255, 0.7);
            }
            QComboBox:focus {
                border: 2px solid rgba(91, 141, 238, 0.8);
                background-color: rgba(255, 255, 255, 0.35);
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid white;
                margin-right: 10px;
            }
            
            /* 下拉列表容器 */
            QComboBox QAbstractItemView {
                background-color: #F0F5FF;
                border: 2px solid #5B8DEE;
                border-radius: 12px;
                padding: 6px;
                outline: none;
                selection-background-color: #5B8DEE;
                selection-color: white;
            }
            
            /* 下拉列表每一项 */
            QComboBox QAbstractItemView::item {
                height: 35px;
                padding-left: 15px;
                border-radius: 8px;
                margin: 3px 4px;
                color: #2c3e50;
                font-size: 11pt;
            }
            
            /* 鼠标悬停效果 */
            QComboBox QAbstractItemView::item:hover {
                background-color: rgba(91, 141, 238, 0.15);
                color: #2c3e50;
            }
            
            /* 选中项效果 */
            QComboBox QAbstractItemView::item:selected {
                background-color: #5B8DEE;
                color: white;
            }
        """)

        port_layout.addWidget(self.port_combo)
        port_layout.addStretch()  # 右侧弹性空间

        control_layout.addWidget(port_container)
        control_layout.addSpacing(100)
        
        # 按钮区域 - 透明容器，只包含按钮
        button_container = QFrame()
        button_container.setStyleSheet("background: transparent;")
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(13)
        
        # 启动演示模式按钮
        self.demo_btn = QPushButton("启动演示模式")
        self.demo_btn.setMinimumHeight(52)
        self.demo_btn.setStyleSheet("""
            QPushButton {
                font-size: 13pt;
                background-color: #5B8DEE;
                color: white;
                border: none;
                border-radius: 26px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #4A7DD9;
            }
            QPushButton:pressed {
                background-color: #3968C2;
            }
        """)
        self.demo_btn.clicked.connect(self.toggle_demo_mode)
        button_layout.addWidget(self.demo_btn)
        
        # 开启演奏模式按钮
        self.play_btn = QPushButton("开启演奏模式")
        self.play_btn.setMinimumHeight(52)
        self.play_btn.setStyleSheet("""
            QPushButton {
                font-size: 13pt;
                background-color: white;
                color: #333;
                border: 2px solid #D0D0D0;
                border-radius: 26px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #F8F8F8;
                border: 2px solid #A0A0A0;
            }
            QPushButton:pressed {
                background-color: #E8E8E8;
            }
        """)
        self.play_btn.clicked.connect(self.toggle_play_mode)
        button_layout.addWidget(self.play_btn)
        
        # 退出程序按钮
        self.exit_btn = QPushButton("退出程序")
        self.exit_btn.setMinimumHeight(52)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                font-size: 13pt;
                background-color: white;
                color: #E74C3C;
                border: 2px solid #D0D0D0;
                border-radius: 26px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #FFF8F8;
                border: 2px solid #E74C3C;
            }
            QPushButton:pressed {
                background-color: #FFE8E8;
            }
        """)
        self.exit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.exit_btn)
        
        control_layout.addWidget(button_container)
        control_layout.addSpacing(220)
        
        # 状态显示区 - 只有状态文本，没有标题和蓝点
        status_container = QFrame()
        status_container.setFixedHeight(140)
        status_container.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border-radius: 18px;
            }
        """)
        status_layout = QVBoxLayout(status_container)
        status_layout.setContentsMargins(15, 15, 15, 15)
        status_layout.setSpacing(0)
        
        self.status_text = QLabel("系统未启动")
        self.status_text.setWordWrap(True)
        self.status_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.status_text.setStyleSheet("""
            color: #333;
            font-size: 10pt;
            padding: 5px;
            background: transparent;
        """)
        status_layout.addWidget(self.status_text)
        
        control_layout.addWidget(status_container)
    
        # 隐藏的help_text用于程序逻辑
        self.help_text = QLabel("")
        self.help_text.hide()
        
        # 添加底部弹性空间
        control_layout.addStretch()
        
        # 音乐可视化（隐藏）
        self.music_viz_label = QLabel()
        self.music_viz_label.setFixedSize(350, 220)
        self.music_viz_label.setStyleSheet("""
            background-color: rgba(42, 42, 42, 0.85);
            border-radius: 15px;
        """)
        self.music_viz_label.hide()
        
        # 如果背景图片存在，调整背景标签大小
        if os.path.exists(bg_path):
            def resize_bg(event):
                bg_label.setGeometry(0, 0, control_frame.width(), control_frame.height())
            control_frame.resizeEvent = resize_bg
        
        main_layout.addWidget(control_frame, 3)
    
    def update_button_style(self):
        """根据运行状态更新按钮样式"""
        if not self.is_running:
            # 开始状态 - 白色
            self.toggle_btn.setText("开始程序")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    font-size: 13pt;
                    background-color: white;
                    color: #333;
                    border: 2px solid #D0D0D0;
                    border-radius: 26px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #F8F8F8;
                    border: 2px solid #A0A0A0;
                }
                QPushButton:pressed {
                    background-color: #E8E8E8;
                }
            """)
        else:
            # 运行状态 - 蓝色
            self.toggle_btn.setText("结束程序")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    font-size: 13pt;
                    background-color: #5B8DEE;
                    color: white;
                    border: none;
                    border-radius: 26px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #4A7DD9;
                }
                QPushButton:pressed {
                    background-color: #3968C2;
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
                    border-radius: 10px;
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
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 10px;
                    border: 2px solid #388E3C;
                }
                QPushButton:hover {
                    background-color: #388E3C;
                    border: 2px solid #2E7D32;
                }
                QPushButton:pressed {
                    background-color: #2E7D32;
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
