"""
渲染引擎模块 - 负责游戏的像素风渲染系统
Renderer Engine Module - Handles pixel-art rendering system for the game
"""

import pygame
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import *


class Renderer:
    """
    渲染引擎类 - 管理所有游戏图形渲染
    Renderer Engine Class - Manages all game graphics rendering
    """
    
    def __init__(self, screen):
        """
        初始化渲染引擎
        Initialize the renderer engine
        
        Args:
            screen: Pygame screen surface
        """
        self.screen = screen
        self.clock = pygame.time.Clock()
        
        # 字体设置
        pygame.font.init()
        self.fonts = {
            'small': pygame.font.Font(None, 24),
            'medium': pygame.font.Font(None, 36),
            'large': pygame.font.Font(None, 48),
            'title': pygame.font.Font(None, 72)
        }
        
        # 渲染缓存
        self.surface_cache = {}
        
        # 像素风设置
        self.pixel_perfect = True
        
        # 调试模式
        self.debug_mode = False
        
    def clear_screen(self, color=COLORS['BLACK']):
        """
        清空屏幕
        Clear the screen
        
        Args:
            color: Background color (default: black)
        """
        self.screen.fill(color)
        
    def draw_pixel_rect(self, color, rect, border_width=0, border_color=None):
        """
        绘制像素完美的矩形
        Draw pixel-perfect rectangle
        
        Args:
            color: Fill color
            rect: Rectangle (x, y, width, height)
            border_width: Border width (default: 0)
            border_color: Border color (default: None)
        """
        # 确保坐标为整数（像素完美）
        x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        pixel_rect = (x, y, w, h)
        
        # 绘制填充
        pygame.draw.rect(self.screen, color, pixel_rect)
        
        # 绘制边框
        if border_width > 0 and border_color:
            pygame.draw.rect(self.screen, border_color, pixel_rect, border_width)
            
    def draw_grid_cell(self, grid_x, grid_y, color, border=False):
        """
        绘制网格单元格
        Draw grid cell
        
        Args:
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
            color: Cell color
            border: Whether to draw border
        """
        x = grid_x * GRID_SIZE
        y = grid_y * GRID_SIZE
        
        cell_rect = (x, y, GRID_SIZE, GRID_SIZE)
        self.draw_pixel_rect(color, cell_rect)
        
        if border:
            border_color = COLORS['DARK_GREEN'] if color == COLORS['GREEN'] else COLORS['GRAY']
            self.draw_pixel_rect(border_color, cell_rect, 1)
            
    def draw_snake(self, snake_segments):
        """
        绘制蛇
        Draw snake
        
        Args:
            snake_segments: List of snake segment positions [(x, y), ...]
        """
        for i, (x, y) in enumerate(snake_segments):
            if i == 0:  # 蛇头
                color = COLORS['DARK_GREEN']
                border = True
            else:  # 蛇身
                color = COLORS['GREEN']
                border = False
                
            self.draw_grid_cell(x, y, color, border)
            
    def draw_food(self, food_pos):
        """
        绘制食物
        Draw food
        
        Args:
            food_pos: Food position (x, y)
        """
        x, y = food_pos
        self.draw_grid_cell(x, y, COLORS['RED'], True)
        
        # 添加食物闪烁效果
        center_x = x * GRID_SIZE + GRID_SIZE // 2
        center_y = y * GRID_SIZE + GRID_SIZE // 2
        pygame.draw.circle(self.screen, COLORS['LIGHT_RED'], 
                         (center_x, center_y), GRID_SIZE // 4)
                         
    def draw_text(self, text, font_size, color, position, center=False):
        """
        绘制文本
        Draw text
        
        Args:
            text: Text to draw
            font_size: Font size ('small', 'medium', 'large', 'title')
            color: Text color
            position: Position (x, y)
            center: Whether to center the text at position
            
        Returns:
            Text rectangle
        """
        font = self.fonts.get(font_size, self.fonts['medium'])
        text_surface = font.render(str(text), False, color)
        
        if center:
            text_rect = text_surface.get_rect(center=position)
            self.screen.blit(text_surface, text_rect)
            return text_rect
        else:
            self.screen.blit(text_surface, position)
            return text_surface.get_rect(topleft=position)
            
    def draw_button(self, text, rect, color, text_color, border_color=None, selected=False):
        """
        绘制按钮
        Draw button
        
        Args:
            text: Button text
            rect: Button rectangle
            color: Button color
            text_color: Text color
            border_color: Border color
            selected: Whether button is selected
            
        Returns:
            Button rectangle
        """
        # 选中状态的颜色调整
        if selected:
            color = tuple(min(255, c + 30) for c in color)
            
        # 绘制按钮背景
        self.draw_pixel_rect(color, rect)
        
        # 绘制边框
        if border_color:
            self.draw_pixel_rect(border_color, rect, 2)
            
        # 绘制文本
        center_x = rect[0] + rect[2] // 2
        center_y = rect[1] + rect[3] // 2
        self.draw_text(text, 'medium', text_color, (center_x, center_y), center=True)
        
        return pygame.Rect(rect)
        
    def draw_game_border(self):
        """
        绘制游戏边框
        Draw game border
        """
        # 计算游戏区域
        game_width = GRID_WIDTH * GRID_SIZE
        game_height = GRID_HEIGHT * GRID_SIZE
        
        # 绘制边框
        border_rect = (-2, -2, game_width + 4, game_height + 4)
        self.draw_pixel_rect(COLORS['WHITE'], border_rect, 2)
        
    def draw_score_panel(self, score, high_score, level="简单"):
        """
        绘制分数面板
        Draw score panel
        
        Args:
            score: Current score
            high_score: High score
            level: Difficulty level
        """
        panel_x = GRID_WIDTH * GRID_SIZE + 20
        panel_y = 20
        
        # 绘制面板背景
        panel_rect = (panel_x, panel_y, 200, 150)
        self.draw_pixel_rect(COLORS['DARK_GRAY'], panel_rect)
        self.draw_pixel_rect(COLORS['WHITE'], panel_rect, 2)
        
        # 绘制分数信息
        y_offset = panel_y + 20
        self.draw_text("分数", 'medium', COLORS['WHITE'], (panel_x + 10, y_offset))
        y_offset += 30
        self.draw_text(str(score), 'large', COLORS['YELLOW'], (panel_x + 10, y_offset))
        
        y_offset += 40
        self.draw_text("最高分", 'medium', COLORS['WHITE'], (panel_x + 10, y_offset))
        y_offset += 30
        self.draw_text(str(high_score), 'medium', COLORS['GREEN'], (panel_x + 10, y_offset))
        
        y_offset += 40
        self.draw_text(f"难度: {level}", 'small', COLORS['LIGHT_GRAY'], (panel_x + 10, y_offset))
        
    def draw_game_over_overlay(self, score, high_score, is_new_record=False):
        """
        绘制游戏结束覆盖层
        Draw game over overlay
        
        Args:
            score: Final score
            high_score: High score
            is_new_record: Whether it's a new record
        """
        # 半透明背景
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(COLORS['BLACK'])
        self.screen.blit(overlay, (0, 0))
        
        # 游戏结束面板
        panel_width = 400
        panel_height = 300
        panel_x = (WINDOW_WIDTH - panel_width) // 2
        panel_y = (WINDOW_HEIGHT - panel_height) // 2
        
        panel_rect = (panel_x, panel_y, panel_width, panel_height)
        self.draw_pixel_rect(COLORS['DARK_GRAY'], panel_rect)
        self.draw_pixel_rect(COLORS['WHITE'], panel_rect, 3)
        
        # 标题
        title_y = panel_y + 30
        self.draw_text("游戏结束", 'title', COLORS['RED'], 
                      (WINDOW_WIDTH // 2, title_y), center=True)
        
        # 分数信息
        score_y = title_y + 80
        self.draw_text(f"得分: {score}", 'large', COLORS['YELLOW'], 
                      (WINDOW_WIDTH // 2, score_y), center=True)
        
        # 最高分
        high_score_y = score_y + 50
        if is_new_record:
            self.draw_text("新纪录!", 'medium', COLORS['GREEN'], 
                          (WINDOW_WIDTH // 2, high_score_y), center=True)
            high_score_y += 30
            
        self.draw_text(f"最高分: {high_score}", 'medium', COLORS['WHITE'], 
                      (WINDOW_WIDTH // 2, high_score_y), center=True)
        
        # 提示信息
        hint_y = panel_y + panel_height - 60
        self.draw_text("按 R 重新开始，按 ESC 返回菜单", 'small', COLORS['LIGHT_GRAY'], 
                      (WINDOW_WIDTH // 2, hint_y), center=True)
                      
    def draw_pause_overlay(self):
        """
        绘制暂停覆盖层
        Draw pause overlay
        """
        # 半透明背景
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(COLORS['BLACK'])
        self.screen.blit(overlay, (0, 0))
        
        # 暂停文本
        self.draw_text("游戏暂停", 'title', COLORS['WHITE'], 
                      (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50), center=True)
        self.draw_text("按空格键继续", 'medium', COLORS['LIGHT_GRAY'], 
                      (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20), center=True)
                      
    def draw_menu_background(self):
        """
        绘制菜单背景
        Draw menu background
        """
        # 渐变背景效果
        for y in range(WINDOW_HEIGHT):
            color_ratio = y / WINDOW_HEIGHT
            r = int(COLORS['DARK_GRAY'][0] * (1 - color_ratio) + COLORS['BLACK'][0] * color_ratio)
            g = int(COLORS['DARK_GRAY'][1] * (1 - color_ratio) + COLORS['BLACK'][1] * color_ratio)
            b = int(COLORS['DARK_GRAY'][2] * (1 - color_ratio) + COLORS['BLACK'][2] * color_ratio)
            
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))
            
    def draw_loading_screen(self, progress=0):
        """
        绘制加载屏幕
        Draw loading screen
        
        Args:
            progress: Loading progress (0-100)
        """
        self.clear_screen()
        
        # 标题
        self.draw_text("贪吃蛇", 'title', COLORS['GREEN'], 
                      (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100), center=True)
        
        # 进度条
        bar_width = 300
        bar_height = 20
        bar_x = (WINDOW_WIDTH - bar_width) // 2
        bar_y = WINDOW_HEIGHT // 2
        
        # 进度条背景
        self.draw_pixel_rect(COLORS['DARK_GRAY'], (bar_x, bar_y, bar_width, bar_height))
        self.draw_pixel_rect(COLORS['WHITE'], (bar_x, bar_y, bar_width, bar_height), 2)
        
        # 进度条填充
        fill_width = int(bar_width * progress / 100)
        if fill_width > 0:
            self.draw_pixel_rect(COLORS['GREEN'], (bar_x, bar_y, fill_width, bar_height))
            
        # 进度文本
        self.draw_text(f"加载中... {progress}%", 'medium', COLORS['WHITE'], 
                      (WINDOW_WIDTH // 2, bar_y + 40), center=True)
                      
    def draw_debug_info(self, fps, snake_length, food_count):
        """
        绘制调试信息
        Draw debug information
        
        Args:
            fps: Current FPS
            snake_length: Snake length
            food_count: Food count
        """
        if not self.debug_mode:
            return
            
        debug_y = 10
        self.draw_text(f"FPS: {fps:.1f}", 'small', COLORS['YELLOW'], (10, debug_y))
        debug_y += 20
        self.draw_text(f"Snake Length: {snake_length}", 'small', COLORS['YELLOW'], (10, debug_y))
        debug_y += 20
        self.draw_text(f"Food Count: {food_count}", 'small', COLORS['YELLOW'], (10, debug_y))
        
    def toggle_debug_mode(self):
        """
        切换调试模式
        Toggle debug mode
        """
        self.debug_mode = not self.debug_mode
        
    def get_fps(self):
        """
        获取当前FPS
        Get current FPS
        
        Returns:
            Current FPS
        """
        return self.clock.get_fps()
        
    def tick(self, fps=FPS):
        """
        更新时钟
        Update clock
        
        Args:
            fps: Target FPS
        """
        self.clock.tick(fps)
        
    def present(self):
        """
        显示渲染结果
        Present rendered frame
        """
        pygame.display.flip()
        
    def cleanup(self):
        """
        清理渲染器资源
        Cleanup renderer resources
        """
        self.surface_cache.clear()
        
    def create_surface(self, width, height, alpha=False):
        """
        创建表面
        Create surface
        
        Args:
            width: Surface width
            height: Surface height
            alpha: Whether to enable alpha
            
        Returns:
            Pygame surface
        """
        if alpha:
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
        else:
            surface = pygame.Surface((width, height))
        return surface
        
    def save_screenshot(self, filename):
        """
        保存屏幕截图
        Save screenshot
        
        Args:
            filename: Screenshot filename
        """
        try:
            pygame.image.save(self.screen, filename)
            return True
        except Exception as e:
            print(f"Failed to save screenshot: {e}")
            return False