import pygame
import sys
import time
import cv2
import numpy as np

# 窗口设置
WIDTH, HEIGHT = 500, 300
FPS = 60

# 音乐数据
my_board = [1, 2, 3, 5, 6]  # 宫商角徵羽对应的音符
my_music = [6, 5, 3, 2, 1, 3, 2, 1, 6, 5, 5, 6, 5, 6, 1, 2, 3, 5, 6, 5, 3, 2, 1, 2]
durations = [0.9, 0.3, 0.6, 0.6, 2.4, 0.9, 0.3, 0.6, 0.6, 2.3, 0.9, 0.3, 0.6, 0.6, 0.9, 0.3, 0.6, 0.6, 0.9, 0.3, 0.6, 0.6, 2.4]

SPEED = HEIGHT / 2.5  # 下落速度(像素/秒)

# 颜色定义
WHITE = (240, 240, 240)
BLACK = (30, 30, 30)
GRAY = (180, 180, 180)
COLOR_PALETTE = [
    (220, 180, 180),  # 浅粉
    (180, 220, 180),  # 浅绿
    (180, 180, 220),  # 浅蓝
    (220, 220, 180),  # 浅黄
    (220, 180, 220),  # 浅紫
]

class Note:
    def __init__(self, note, duration, speed, start_time, index):
        self.note = note
        self.duration = duration
        self.width = WIDTH // 5
        self.height = speed * duration
        self.speed = speed
        
        # 计算x位置
        note_index = my_board.index(note) if note in my_board else 0
        self.x = note_index * self.width
        
        # 计算y位置 (初始位置在屏幕上方)
        self.start_y = -self.height - (start_time * speed)
        self.y = self.start_y
        
        # 音符颜色
        self.color = COLOR_PALETTE[note % len(COLOR_PALETTE)]
        
        # 音符状态
        self.active = True
        self.passed = False  # 是否已经通过键盘区域
    
    def update(self, current_time):
        """更新音符位置"""
        self.y = self.start_y + (current_time * self.speed)
        
        # 检查是否超出屏幕
        if self.y > HEIGHT:
            self.active = False
        
        # 检查是否通过键盘区域
        if not self.passed and self.y >= HEIGHT - (HEIGHT // 5) - self.height:
            self.passed = True
            return True  # 返回True表示应该播放音符
        return False
    
    def is_visible(self):
        """检查音符是否在可见范围内"""
        return self.y + self.height > 0 and self.y < HEIGHT
    
    def draw(self, screen):
        """绘制音符"""
        if self.is_visible():
            pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
            # 绘制音符数字
            font = pygame.font.SysFont('SimHei', 20)
            text = font.render(str(self.note), True, BLACK)
            screen.blit(text, (self.x + self.width//2 - 5, self.y + self.height//2 - 10))

def draw_musical_notes(screen, width, height, alpha=10):
    """在屏幕上绘制五等分的宫商角徵羽"""
    section_width = width // 5
    notes = ["小指", "无名", "中指", "食指", "拇指"]
    font = pygame.font.SysFont('SimHei', 30)
    
    # 创建半透明表面
    note_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    
    for i in range(5):
        text = font.render(notes[i], True, (150, 150, 150, alpha))
        text_rect = text.get_rect(center=(section_width * (i + 0.5), height // 2))
        note_surface.blit(text, text_rect)
    
    screen.blit(note_surface, (0, 0))

def draw_keyboard(screen, width, height, active_keys=None):
    """在屏幕底部1/5高度绘制键盘"""
    if active_keys is None:
        active_keys = set()
    
    keyboard_height = height // 5
    keyboard_top = height - keyboard_height
    
    # 绘制键盘背景
    pygame.draw.rect(screen, GRAY, (0, keyboard_top, width, keyboard_height))
    
    # 五等分绘制琴键
    key_width = width // 5
    for i in range(5):
        key_left = i * key_width
        note = my_board[i] if i < len(my_board) else 0
        
        # 如果键被激活，颜色变亮
        color = COLOR_PALETTE[note % len(COLOR_PALETTE)] if note in active_keys else WHITE
        
        pygame.draw.rect(screen, color, (key_left, keyboard_top, key_width-2, keyboard_height))
        pygame.draw.rect(screen, BLACK, (key_left, keyboard_top, key_width-2, keyboard_height), 1)
        
        # 添加音阶标签
        font = pygame.font.SysFont('SimHei', 20)
        note_labels = ["宫", "商", "角", "徵", "羽"]
        label = font.render(note_labels[i], True, BLACK)
        screen.blit(label, (key_left + key_width//2 - 10, keyboard_top + keyboard_height//2 - 10))

def get_frame_generator():
    # 初始化 Pygame
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    def main_loop():
        # 预计算所有音符
        all_notes = []
        current_time = 0.0
        for i, (note, duration) in enumerate(zip(my_music, durations)):
            all_notes.append(Note(note, duration, SPEED, current_time, i))
            current_time += duration
        
        # 游戏主循环
        running = True
        start_time = time.time()
        active_keys = set()
        key_active_times = {}
        
        while running:
            current_time = time.time() - start_time
            
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # 更新音符状态和检测激活键
            active_keys.clear()
            for note in all_notes:
                if note.update(current_time):
                    active_keys.add(note.note)
                    key_active_times[note.note] = current_time + 0.3  # 高亮0.3秒
            
            # 更新键盘高亮状态
            keys_to_remove = []
            for note, end_time in key_active_times.items():
                if current_time > end_time:
                    keys_to_remove.append(note)
            for note in keys_to_remove:
                del key_active_times[note]
            
            # 合并当前激活键和键盘高亮键
            display_active_keys = set(active_keys)
            display_active_keys.update(key_active_times.keys())
            
            # 渲染
            screen.fill(WHITE)
            
            # 绘制五等分的宫商角徵羽
            draw_musical_notes(screen, WIDTH, HEIGHT)
            
            # 绘制下落音符 (只绘制可见的)
            for note in all_notes:
                if note.is_visible():
                    note.draw(screen)
            
            # 绘制键盘
            draw_keyboard(screen, WIDTH, HEIGHT, display_active_keys)
            
            # 显示当前时间
            font = pygame.font.SysFont('SimHei', 20)
            time_text = f"时间: {current_time:.2f}s"
            text = font.render(time_text, True, BLACK)
            screen.blit(text, (10, 10))
            
            # 转换为RGB格式并旋转90度
            frame = pygame.surfarray.array3d(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 旋转90度并调整大小
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # 水平镜像
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (500, 300))  # 匹配控制面板中的窗口大小
            yield frame
            
            clock.tick(FPS)
        
        pygame.quit()
    
    return main_loop()

if __name__ == "__main__":
    # 独立运行时的演示模式
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("音乐可视化播放器")
    
    for frame in get_frame_generator():
        # 显示帧
        pygame.surfarray.blit_array(screen, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pygame.display.flip()
        
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
