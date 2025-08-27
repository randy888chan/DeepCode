"""
事件处理器 (Event Handler)
处理所有用户输入和游戏事件
"""

import pygame
import sys
from typing import Dict, List, Callable, Optional, Any
from config.constants import *

class EventHandler:
    """事件处理器类 - 管理所有用户输入和游戏事件"""
    
    def __init__(self):
        """初始化事件处理器"""
        self.key_states = {}  # 按键状态记录
        self.mouse_pos = (0, 0)  # 鼠标位置
        self.mouse_buttons = {}  # 鼠标按键状态
        self.event_callbacks = {}  # 事件回调函数
        self.key_repeat_delay = 200  # 按键重复延迟(毫秒)
        self.key_repeat_interval = 50  # 按键重复间隔(毫秒)
        self.last_key_time = {}  # 上次按键时间
        self.quit_requested = False  # 退出请求标志
        
        # 初始化按键状态
        self._init_key_states()
        
        # 设置按键重复
        pygame.key.set_repeat(self.key_repeat_delay, self.key_repeat_interval)
        
    def _init_key_states(self):
        """初始化按键状态"""
        # 方向键
        for key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                   pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
            self.key_states[key] = False
            self.last_key_time[key] = 0
            
        # 功能键
        for key in [pygame.K_SPACE, pygame.K_RETURN, pygame.K_ESCAPE,
                   pygame.K_p, pygame.K_r, pygame.K_q, pygame.K_F1]:
            self.key_states[key] = False
            self.last_key_time[key] = 0
            
        # 鼠标按键
        for button in [1, 2, 3]:  # 左键、中键、右键
            self.mouse_buttons[button] = False
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册事件回调函数
        
        Args:
            event_type: 事件类型 ('key_down', 'key_up', 'mouse_down', 'mouse_up', 'mouse_motion', 'quit')
            callback: 回调函数
        """
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable):
        """取消注册事件回调函数"""
        if event_type in self.event_callbacks:
            if callback in self.event_callbacks[event_type]:
                self.event_callbacks[event_type].remove(callback)
    
    def _trigger_callbacks(self, event_type: str, event_data: Any = None):
        """触发事件回调函数"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if event_data is not None:
                        callback(event_data)
                    else:
                        callback()
                except Exception as e:
                    print(f"事件回调错误: {e}")
    
    def handle_events(self) -> List[pygame.event.Event]:
        """处理所有事件并返回事件列表"""
        events = pygame.event.get()
        current_time = pygame.time.get_ticks()
        
        for event in events:
            self._process_event(event, current_time)
        
        # 更新鼠标位置
        self.mouse_pos = pygame.mouse.get_pos()
        
        return events
    
    def _process_event(self, event: pygame.event.Event, current_time: int):
        """处理单个事件"""
        if event.type == pygame.QUIT:
            self.quit_requested = True
            self._trigger_callbacks('quit', event)
            
        elif event.type == pygame.KEYDOWN:
            self.key_states[event.key] = True
            self.last_key_time[event.key] = current_time
            self._trigger_callbacks('key_down', event)
            
        elif event.type == pygame.KEYUP:
            self.key_states[event.key] = False
            self._trigger_callbacks('key_up', event)
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_buttons[event.button] = True
            self._trigger_callbacks('mouse_down', event)
            
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_buttons[event.button] = False
            self._trigger_callbacks('mouse_up', event)
            
        elif event.type == pygame.MOUSEMOTION:
            self._trigger_callbacks('mouse_motion', event)
    
    def is_key_pressed(self, key: int) -> bool:
        """检查按键是否被按下"""
        return self.key_states.get(key, False)
    
    def is_key_just_pressed(self, key: int, current_time: int = None) -> bool:
        """检查按键是否刚刚被按下(防止重复触发)"""
        if current_time is None:
            current_time = pygame.time.get_ticks()
            
        if not self.is_key_pressed(key):
            return False
            
        # 检查是否在重复延迟时间内
        time_since_press = current_time - self.last_key_time.get(key, 0)
        return time_since_press < self.key_repeat_delay
    
    def is_mouse_button_pressed(self, button: int) -> bool:
        """检查鼠标按键是否被按下"""
        return self.mouse_buttons.get(button, False)
    
    def get_mouse_pos(self) -> tuple:
        """获取鼠标位置"""
        return self.mouse_pos
    
    def get_grid_pos_from_mouse(self) -> tuple:
        """将鼠标位置转换为网格坐标"""
        mouse_x, mouse_y = self.mouse_pos
        
        # 考虑游戏区域偏移
        game_x = mouse_x - GAME_AREA_X
        game_y = mouse_y - GAME_AREA_Y
        
        # 转换为网格坐标
        if 0 <= game_x < GAME_AREA_WIDTH and 0 <= game_y < GAME_AREA_HEIGHT:
            grid_x = game_x // CELL_SIZE
            grid_y = game_y // CELL_SIZE
            return (grid_x, grid_y)
        
        return None
    
    def is_quit_requested(self) -> bool:
        """检查是否请求退出"""
        return self.quit_requested
    
    def reset_quit_request(self):
        """重置退出请求"""
        self.quit_requested = False
    
    def get_direction_input(self) -> Optional[str]:
        """获取方向输入"""
        # 检查方向键
        if self.is_key_pressed(pygame.K_UP) or self.is_key_pressed(pygame.K_w):
            return DIRECTIONS['UP']
        elif self.is_key_pressed(pygame.K_DOWN) or self.is_key_pressed(pygame.K_s):
            return DIRECTIONS['DOWN']
        elif self.is_key_pressed(pygame.K_LEFT) or self.is_key_pressed(pygame.K_a):
            return DIRECTIONS['LEFT']
        elif self.is_key_pressed(pygame.K_RIGHT) or self.is_key_pressed(pygame.K_d):
            return DIRECTIONS['RIGHT']
        
        return None
    
    def is_pause_pressed(self) -> bool:
        """检查是否按下暂停键"""
        return self.is_key_just_pressed(pygame.K_SPACE) or self.is_key_just_pressed(pygame.K_p)
    
    def is_restart_pressed(self) -> bool:
        """检查是否按下重新开始键"""
        return self.is_key_just_pressed(pygame.K_r)
    
    def is_menu_pressed(self) -> bool:
        """检查是否按下菜单键"""
        return self.is_key_just_pressed(pygame.K_ESCAPE)
    
    def is_confirm_pressed(self) -> bool:
        """检查是否按下确认键"""
        return self.is_key_just_pressed(pygame.K_RETURN) or self.is_key_just_pressed(pygame.K_SPACE)
    
    def is_debug_pressed(self) -> bool:
        """检查是否按下调试键"""
        return self.is_key_just_pressed(pygame.K_F1)
    
    def check_button_click(self, button_rect: pygame.Rect) -> bool:
        """检查按钮是否被点击"""
        if self.is_mouse_button_pressed(1):  # 左键
            return button_rect.collidepoint(self.mouse_pos)
        return False
    
    def is_mouse_over_button(self, button_rect: pygame.Rect) -> bool:
        """检查鼠标是否悬停在按钮上"""
        return button_rect.collidepoint(self.mouse_pos)
    
    def get_menu_navigation(self) -> Optional[str]:
        """获取菜单导航输入"""
        if self.is_key_just_pressed(pygame.K_UP) or self.is_key_just_pressed(pygame.K_w):
            return 'up'
        elif self.is_key_just_pressed(pygame.K_DOWN) or self.is_key_just_pressed(pygame.K_s):
            return 'down'
        elif self.is_key_just_pressed(pygame.K_LEFT) or self.is_key_just_pressed(pygame.K_a):
            return 'left'
        elif self.is_key_just_pressed(pygame.K_RIGHT) or self.is_key_just_pressed(pygame.K_d):
            return 'right'
        elif self.is_key_just_pressed(pygame.K_RETURN) or self.is_key_just_pressed(pygame.K_SPACE):
            return 'select'
        elif self.is_key_just_pressed(pygame.K_ESCAPE):
            return 'back'
        
        return None
    
    def handle_text_input(self, event: pygame.event.Event) -> Optional[str]:
        """处理文本输入"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                return 'backspace'
            elif event.key == pygame.K_RETURN:
                return 'enter'
            elif event.key == pygame.K_ESCAPE:
                return 'escape'
            elif event.unicode and event.unicode.isprintable():
                return event.unicode
        
        return None
    
    def set_key_repeat(self, delay: int, interval: int):
        """设置按键重复参数"""
        self.key_repeat_delay = delay
        self.key_repeat_interval = interval
        pygame.key.set_repeat(delay, interval)
    
    def disable_key_repeat(self):
        """禁用按键重复"""
        pygame.key.set_repeat()
    
    def enable_key_repeat(self):
        """启用按键重复"""
        pygame.key.set_repeat(self.key_repeat_delay, self.key_repeat_interval)
    
    def clear_key_states(self):
        """清除所有按键状态"""
        for key in self.key_states:
            self.key_states[key] = False
        for button in self.mouse_buttons:
            self.mouse_buttons[button] = False
    
    def get_pressed_keys(self) -> List[int]:
        """获取当前按下的所有按键"""
        return [key for key, pressed in self.key_states.items() if pressed]
    
    def get_key_name(self, key: int) -> str:
        """获取按键名称"""
        try:
            return pygame.key.name(key)
        except:
            return f"Key_{key}"
    
    def is_modifier_pressed(self) -> Dict[str, bool]:
        """检查修饰键状态"""
        mods = pygame.key.get_mods()
        return {
            'shift': bool(mods & pygame.KMOD_SHIFT),
            'ctrl': bool(mods & pygame.KMOD_CTRL),
            'alt': bool(mods & pygame.KMOD_ALT),
            'meta': bool(mods & pygame.KMOD_META)
        }
    
    def handle_window_events(self, event: pygame.event.Event) -> Optional[str]:
        """处理窗口事件"""
        if event.type == pygame.VIDEORESIZE:
            return 'resize'
        elif event.type == pygame.VIDEOEXPOSE:
            return 'expose'
        elif event.type == pygame.ACTIVEEVENT:
            if event.gain:
                return 'focus_gained'
            else:
                return 'focus_lost'
        
        return None
    
    def get_joystick_input(self) -> Dict[str, Any]:
        """获取手柄输入(如果有)"""
        joystick_data = {
            'connected': False,
            'axes': [],
            'buttons': [],
            'direction': None
        }
        
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            if not joystick.get_init():
                joystick.init()
            
            joystick_data['connected'] = True
            
            # 获取轴数据
            for i in range(joystick.get_numaxes()):
                joystick_data['axes'].append(joystick.get_axis(i))
            
            # 获取按钮数据
            for i in range(joystick.get_numbuttons()):
                joystick_data['buttons'].append(joystick.get_button(i))
            
            # 解析方向
            if len(joystick_data['axes']) >= 2:
                x_axis = joystick_data['axes'][0]
                y_axis = joystick_data['axes'][1]
                
                if abs(x_axis) > 0.5 or abs(y_axis) > 0.5:
                    if abs(x_axis) > abs(y_axis):
                        joystick_data['direction'] = 'right' if x_axis > 0 else 'left'
                    else:
                        joystick_data['direction'] = 'down' if y_axis > 0 else 'up'
        
        return joystick_data
    
    def update(self):
        """更新事件处理器状态"""
        # 处理事件
        self.handle_events()
        
        # 更新按键状态
        pressed_keys = pygame.key.get_pressed()
        for key in self.key_states:
            if key < len(pressed_keys):
                self.key_states[key] = pressed_keys[key]
        
        # 更新鼠标状态
        mouse_buttons = pygame.mouse.get_pressed()
        for i, pressed in enumerate(mouse_buttons):
            self.mouse_buttons[i + 1] = pressed
    
    def cleanup(self):
        """清理事件处理器资源"""
        self.event_callbacks.clear()
        self.clear_key_states()
        self.quit_requested = False
        
        # 重置按键重复
        pygame.key.set_repeat()
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        return {
            'pressed_keys': self.get_pressed_keys(),
            'mouse_pos': self.mouse_pos,
            'mouse_buttons': [btn for btn, pressed in self.mouse_buttons.items() if pressed],
            'quit_requested': self.quit_requested,
            'callbacks_count': {event_type: len(callbacks) for event_type, callbacks in self.event_callbacks.items()},
            'modifiers': self.is_modifier_pressed()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"EventHandler(keys={len([k for k, v in self.key_states.items() if v])}, mouse={self.mouse_pos})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"EventHandler(pressed_keys={self.get_pressed_keys()}, mouse_pos={self.mouse_pos}, quit={self.quit_requested})"