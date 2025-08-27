#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贪吃蛇桌面游戏 - 主程序入口
Snake Desktop Game - Main Entry Point

经典像素风贪吃蛇桌面游戏，支持多难度、本地存档和流畅操作体验
Classic pixel-style Snake desktop game with multiple difficulties, local saves, and smooth gameplay
"""

import sys
import os
import pygame
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入游戏模块
try:
    from config.constants import *
    from ui.scenes import SceneManager
    from engine.renderer import Renderer
    from engine.event_handler import EventHandler
    from engine.sound_manager import SoundManager
    from data.settings import SettingsManager
    from data.score_manager import ScoreManager
except ImportError as e:
    print(f"导入模块失败 (Import failed): {e}")
    print("请确保所有必要的模块都已正确安装 (Please ensure all required modules are properly installed)")
    sys.exit(1)


class SnakeGame:
    """
    贪吃蛇游戏主类
    Main Snake Game Class
    """
    
    def __init__(self):
        """初始化游戏"""
        self.running = False
        self.clock = None
        self.screen = None
        
        # 游戏组件
        self.renderer = None
        self.event_handler = None
        self.sound_manager = None
        self.scene_manager = None
        self.settings_manager = None
        self.score_manager = None
        
        # 初始化Pygame
        self._init_pygame()
        
        # 初始化游戏组件
        self._init_components()
        
    def _init_pygame(self):
        """初始化Pygame系统"""
        try:
            pygame.init()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # 创建游戏窗口
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption(GAME_TITLE)
            
            # 设置游戏图标（如果存在）
            icon_path = os.path.join(IMAGES_PATH, "icon.png")
            if os.path.exists(icon_path):
                icon = pygame.image.load(icon_path)
                pygame.display.set_icon(icon)
            
            # 创建时钟对象
            self.clock = pygame.time.Clock()
            
            print(f"Pygame初始化成功 - 窗口大小: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
            
        except pygame.error as e:
            print(f"Pygame初始化失败: {e}")
            sys.exit(1)
    
    def _init_components(self):
        """初始化游戏组件"""
        try:
            # 初始化数据管理器
            self.settings_manager = SettingsManager()
            self.score_manager = ScoreManager()
            
            # 初始化引擎组件
            self.renderer = Renderer(self.screen)
            self.event_handler = EventHandler()
            self.sound_manager = SoundManager()
            
            # 初始化场景管理器
            self.scene_manager = SceneManager(
                renderer=self.renderer,
                event_handler=self.event_handler,
                sound_manager=self.sound_manager,
                settings_manager=self.settings_manager,
                score_manager=self.score_manager
            )
            
            print("游戏组件初始化成功")
            
        except Exception as e:
            print(f"游戏组件初始化失败: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def run(self):
        """运行游戏主循环"""
        print("开始运行贪吃蛇游戏...")
        self.running = True
        
        try:
            while self.running:
                # 计算帧时间
                dt = self.clock.tick(FPS) / 1000.0
                
                # 处理事件
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        self.running = False
                    else:
                        # 将事件传递给事件处理器
                        self.event_handler.handle_event(event)
                
                # 检查是否需要退出游戏
                if self.event_handler.should_quit():
                    self.running = False
                
                # 更新场景
                self.scene_manager.update(dt)
                
                # 渲染场景
                self.scene_manager.render()
                
                # 更新显示
                pygame.display.flip()
                
        except KeyboardInterrupt:
            print("\n游戏被用户中断")
        except Exception as e:
            print(f"游戏运行时发生错误: {e}")
            traceback.print_exc()
        finally:
            self.quit()
    
    def quit(self):
        """退出游戏"""
        print("正在退出游戏...")
        
        # 保存设置和分数
        if self.settings_manager:
            self.settings_manager.save_settings()
        if self.score_manager:
            self.score_manager.save_scores()
        
        # 停止音效
        if self.sound_manager:
            self.sound_manager.stop_all()
        
        # 退出Pygame
        pygame.mixer.quit()
        pygame.quit()
        
        print("游戏已退出")


def main():
    """主函数"""
    print("=" * 50)
    print("🐍 贪吃蛇桌面游戏 Snake Desktop Game 🐍")
    print("=" * 50)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # 检查Pygame版本
    try:
        pygame_version = pygame.version.ver
        print(f"Python版本: {sys.version}")
        print(f"Pygame版本: {pygame_version}")
        
        if tuple(map(int, pygame_version.split('.'))) < (2, 0, 0):
            print("警告: 建议使用Pygame 2.0.0或更高版本")
            print("Warning: Pygame 2.0.0 or higher is recommended")
    except:
        print("无法检测Pygame版本")
    
    # 创建并运行游戏
    try:
        game = SnakeGame()
        game.run()
    except Exception as e:
        print(f"游戏启动失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()