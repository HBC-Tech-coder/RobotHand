import sys
import serial
from serial.tools import list_ports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, 
                            QComboBox, QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class SerialVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.received_data = []
        
        # 初始化UI
        self.setWindowTitle("串口通信可视化")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建主窗口和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QHBoxLayout(main_widget)
        
        # 初始化UI组件
        self.init_serial_panel()
        self.init_visual_panel()
        
    def init_serial_panel(self):
        """初始化串口控制面板"""
        serial_panel = QWidget()
        serial_layout = QVBoxLayout(serial_panel)
        
        # 串口控制区域
        serial_group = QGroupBox("串口设置")
        group_layout = QVBoxLayout()
        
        self.port_combo = QComboBox()
        self.refresh_ports()
        
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        group_layout.addWidget(QLabel("选择串口:"))
        group_layout.addWidget(self.port_combo)
        group_layout.addWidget(self.connect_btn)
        serial_group.setLayout(group_layout)
        serial_layout.addWidget(serial_group)
        
        # 数据发送区域
        send_group = QGroupBox("数据发送")
        send_layout = QVBoxLayout()
        
        self.send_input = QTextEdit()
        self.send_input.setMaximumHeight(100)
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send_data)
        
        send_layout.addWidget(QLabel("发送内容:"))
        send_layout.addWidget(self.send_input)
        send_layout.addWidget(self.send_btn)
        send_group.setLayout(send_layout)
        serial_layout.addWidget(send_group)
        
        self.main_layout.addWidget(serial_panel, 1)
    
    def init_visual_panel(self):
        """初始化可视化面板"""
        visual_panel = QWidget()
        visual_layout = QVBoxLayout(visual_panel)
        
        # 可视化区域
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        visual_layout.addWidget(self.canvas)
        
        # 日志区域
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        visual_layout.addWidget(self.log)
        
        self.main_layout.addWidget(visual_panel, 2)
    
    def refresh_ports(self):
        """刷新可用串口列表"""
        self.port_combo.clear()
        ports = list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)
    
    def toggle_connection(self):
        """切换串口连接状态"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.serial_port = None
            self.connect_btn.setText("连接")
            self.log_message("串口已断开")
        else:
            port = self.port_combo.currentText()
            if not port:
                self.log_message("错误：请选择串口")
                return
            
            try:
                self.serial_port = serial.Serial(port, 9600, timeout=1)
                self.connect_btn.setText("断开")
                self.log_message(f"已连接到 {port}")
            except Exception as e:
                self.log_message(f"连接失败: {str(e)}")
    
    def send_data(self):
        """发送串口数据"""
        if not (self.serial_port and self.serial_port.is_open):
            self.log_message("错误：请先连接串口")
            return
        
        data = self.send_input.toPlainText()
        if not data:
            self.log_message("错误：请输入要发送的内容")
            return
        
        try:
            self.serial_port.write(data.encode())
            self.log_message(f"发送: {data}")
            self.update_plot()
        except Exception as e:
            self.log_message(f"发送失败: {str(e)}")
    
    def update_plot(self):
        """更新可视化图表"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 简单示例: 显示最近10次数据发送记录
        ax.plot([1,2,3,4,5,6,7,8,9,10], [10,20,15,25,30,20,15,10,5,0])
        ax.set_xlabel('时间')
        ax.set_ylabel('数据量')
        ax.set_title('串口通信数据')
        
        self.canvas.draw()
    
    def log_message(self, message):
        """记录日志信息"""
        self.log.append(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SerialVisualizer()
    window.show()
    sys.exit(app.exec_())
