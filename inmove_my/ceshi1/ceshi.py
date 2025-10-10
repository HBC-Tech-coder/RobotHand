import sys
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QComboBox, QLabel, 
                             QGroupBox, QCheckBox, QTextEdit, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal

class SerialThread(QThread):
    data_received = pyqtSignal(str)
    connection_status = pyqtSignal(bool)
    
    def __init__(self, port=None, baudrate=9600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        
    def run(self):
        if not self.port:
            self.connection_status.emit(False)
            return
            
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            self.connection_status.emit(True)
            self.running = True
            
            while self.running:
                if self.serial_conn.in_waiting:
                    data = self.serial_conn.readline().decode('utf-8').strip()
                    if data:
                        self.data_received.emit(data)
                        
        except Exception as e:
            self.connection_status.emit(False)
            self.data_received.emit(f"Error: {str(e)}")
            
    def stop(self):
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.wait()
        
    def send_data(self, data):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write((data + '\n').encode('utf-8'))
            except Exception as e:
                self.data_received.emit(f"Send Error: {str(e)}")

class HandControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械手控制工具")
        self.setGeometry(100, 100, 600, 500)
        
        self.serial_thread = None
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        
        # 串口设置区域
        serial_group = QGroupBox("串口设置")
        serial_layout = QHBoxLayout()
        
        self.port_combo = QComboBox()
        self.refresh_ports()
        
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baudrate_combo.setCurrentText("9600")
        
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        self.status_label = QLabel("状态: 未连接")
        
        serial_layout.addWidget(QLabel("端口:"))
        serial_layout.addWidget(self.port_combo)
        serial_layout.addWidget(QLabel("波特率:"))
        serial_layout.addWidget(self.baudrate_combo)
        serial_layout.addWidget(self.connect_btn)
        serial_layout.addWidget(self.status_label)
        serial_group.setLayout(serial_layout)
        
        # 手指控制区域
        finger_group = QGroupBox("手指控制 (1=弯曲, 0=伸直)")
        finger_layout = QVBoxLayout()
        
        # 手腕控制
        wrist_layout = QHBoxLayout()
        self.wrist_check = QCheckBox("手腕")
        wrist_layout.addWidget(self.wrist_check)
        
        # 手指控制
        fingers_layout = QHBoxLayout()
        self.index_check = QCheckBox("食指")
        self.middle_check = QCheckBox("中指")
        self.ring_check = QCheckBox("无名指")
        self.thumb_check = QCheckBox("拇指")
        self.pinky_check = QCheckBox("小指")
        
        fingers_layout.addWidget(self.index_check)
        fingers_layout.addWidget(self.middle_check)
        fingers_layout.addWidget(self.ring_check)
        fingers_layout.addWidget(self.thumb_check)
        fingers_layout.addWidget(self.pinky_check)
        
        # 动作速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("动作速度:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(15)
        speed_layout.addWidget(self.speed_spin)
        
        # 发送按钮
        self.send_btn = QPushButton("发送手势")
        self.send_btn.clicked.connect(self.send_gesture)
        self.send_btn.setEnabled(False)
        
        # 预设手势按钮
        preset_layout = QHBoxLayout()
        self.fist_btn = QPushButton("握拳 (111111)")
        self.open_btn = QPushButton("张开 (000000)")
        self.point_btn = QPushButton("指向 (010000)")
        self.ok_btn = QPushButton("OK手势 (001100)")
        
        self.fist_btn.clicked.connect(lambda: self.set_preset("111111"))
        self.open_btn.clicked.connect(lambda: self.set_preset("000000"))
        self.point_btn.clicked.connect(lambda: self.set_preset("010000"))
        self.ok_btn.clicked.connect(lambda: self.set_preset("001100"))
        
        preset_layout.addWidget(self.fist_btn)
        preset_layout.addWidget(self.open_btn)
        preset_layout.addWidget(self.point_btn)
        preset_layout.addWidget(self.ok_btn)
        
        finger_layout.addLayout(wrist_layout)
        finger_layout.addLayout(fingers_layout)
        finger_layout.addLayout(speed_layout)
        finger_layout.addWidget(self.send_btn)
        finger_layout.addLayout(preset_layout)
        finger_group.setLayout(finger_layout)
        
        # 日志区域
        log_group = QGroupBox("通信日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.clear_log)
        
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(self.clear_log_btn)
        log_group.setLayout(log_layout)
        
        # 添加到主布局
        main_layout.addWidget(serial_group)
        main_layout.addWidget(finger_group)
        main_layout.addWidget(log_group)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)
            
    def toggle_connection(self):
        if self.serial_thread and self.serial_thread.isRunning():
            self.disconnect_serial()
        else:
            self.connect_serial()
            
    def connect_serial(self):
        port = self.port_combo.currentText()
        baudrate = int(self.baudrate_combo.currentText())
        
        if not port:
            self.log_text.append("错误: 没有选择串口")
            return
            
        self.serial_thread = SerialThread(port, baudrate)
        self.serial_thread.data_received.connect(self.handle_received_data)
        self.serial_thread.connection_status.connect(self.update_connection_status)
        self.serial_thread.start()
        
        self.connect_btn.setText("断开")
        self.send_btn.setEnabled(True)
        
    def disconnect_serial(self):
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None
            
        self.connect_btn.setText("连接")
        self.status_label.setText("状态: 未连接")
        self.send_btn.setEnabled(False)
        
    def update_connection_status(self, connected):
        if connected:
            self.status_label.setText("状态: 已连接")
            self.log_text.append(f"已连接到 {self.port_combo.currentText()}")
        else:
            self.status_label.setText("状态: 连接失败")
            self.log_text.append("连接失败")
            
    def handle_received_data(self, data):
        self.log_text.append(f"接收: {data}")
        
    def send_gesture(self):
        # 构建6位二进制字符串 (顺序: 手腕, 食指, 中指, 无名指, 拇指, 小指)
        gesture = (
            ("1" if self.wrist_check.isChecked() else "0") +
            ("1" if self.index_check.isChecked() else "0") +
            ("1" if self.middle_check.isChecked() else "0") +
            ("1" if self.ring_check.isChecked() else "0") +
            ("1" if self.thumb_check.isChecked() else "0") +
            ("1" if self.pinky_check.isChecked() else "0")
        )
        
        # 发送速度参数 (可选)
        speed = self.speed_spin.value()
        # 可以在这里将速度参数添加到发送数据中，如果需要
        
        if self.serial_thread:
            self.serial_thread.send_data(gesture)
            self.log_text.append(f"发送: {gesture} (速度: {speed})")
            
    def set_preset(self, gesture):
        # 设置预设手势 (6位二进制字符串)
        if len(gesture) != 6:
            return
            
        # 更新复选框状态
        self.wrist_check.setChecked(gesture[0] == '1')
        self.index_check.setChecked(gesture[1] == '1')
        self.middle_check.setChecked(gesture[2] == '1')
        self.ring_check.setChecked(gesture[3] == '1')
        self.thumb_check.setChecked(gesture[4] == '1')
        self.pinky_check.setChecked(gesture[5] == '1')
        
    def clear_log(self):
        self.log_text.clear()
        
    def closeEvent(self, event):
        self.disconnect_serial()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandControlApp()
    window.show()
    sys.exit(app.exec_())