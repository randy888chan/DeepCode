"""
Visual Effects and Animations Module for Snake Game

This module provides various visual effects including:
- Particle systems for food consumption and game events
- Animation utilities for smooth transitions
- Visual feedback effects for score changes
- Background effects and gradients
- Screen transitions and fade effects
"""

import pygame
import math
import random
import time
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

from ..utils.constants import *


class ParticleType(Enum):
    """Types of particles for different effects"""
    FOOD_CONSUME = "food_consume"
    SCORE_POPUP = "score_popup"
    EXPLOSION = "explosion"
    SPARKLE = "sparkle"
    TRAIL = "trail"


class AnimationType(Enum):
    """Types of animations"""
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SLIDE_IN = "slide_in"
    SLIDE_OUT = "slide_out"
    BOUNCE = "bounce"
    PULSE = "pulse"


class EasingFunction:
    """Easing functions for smooth animations"""
    
    @staticmethod
    def linear(t: float) -> float:
        """Linear interpolation"""
        return t
    
    @staticmethod
    def ease_in_quad(t: float) -> float:
        """Quadratic ease-in"""
        return t * t
    
    @staticmethod
    def ease_out_quad(t: float) -> float:
        """Quadratic ease-out"""
        return 1 - (1 - t) * (1 - t)
    
    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        """Quadratic ease-in-out"""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2
    
    @staticmethod
    def ease_in_cubic(t: float) -> float:
        """Cubic ease-in"""
        return t * t * t
    
    @staticmethod
    def ease_out_cubic(t: float) -> float:
        """Cubic ease-out"""
        return 1 - pow(1 - t, 3)
    
    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out"""
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2
    
    @staticmethod
    def bounce_out(t: float) -> float:
        """Bounce ease-out"""
        n1 = 7.5625
        d1 = 2.75
        
        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375


class Particle:
    """Individual particle for particle effects"""
    
    def __init__(self, x: float, y: float, particle_type: ParticleType, **kwargs):
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.particle_type = particle_type
        
        # Movement properties
        self.velocity_x = kwargs.get('velocity_x', random.uniform(-2, 2))
        self.velocity_y = kwargs.get('velocity_y', random.uniform(-2, 2))
        self.acceleration_x = kwargs.get('acceleration_x', 0)
        self.acceleration_y = kwargs.get('acceleration_y', 0.1)  # Gravity
        
        # Visual properties
        self.color = kwargs.get('color', WHITE)
        self.size = kwargs.get('size', random.uniform(2, 6))
        self.start_size = self.size
        self.alpha = kwargs.get('alpha', 255)
        self.start_alpha = self.alpha
        
        # Lifecycle properties
        self.lifetime = kwargs.get('lifetime', 1.0)  # seconds
        self.age = 0.0
        self.is_alive = True
        
        # Animation properties
        self.rotation = kwargs.get('rotation', 0)
        self.rotation_speed = kwargs.get('rotation_speed', random.uniform(-5, 5))
        self.scale_factor = kwargs.get('scale_factor', 1.0)
        
        # Special properties based on type
        if particle_type == ParticleType.SCORE_POPUP:
            self.text = kwargs.get('text', '+10')
            self.font_size = kwargs.get('font_size', FONT_SIZE_MEDIUM)
            self.velocity_y = -1.5  # Move upward
            self.acceleration_y = 0  # No gravity for text
        elif particle_type == ParticleType.SPARKLE:
            self.twinkle_speed = kwargs.get('twinkle_speed', 5.0)
        elif particle_type == ParticleType.EXPLOSION:
            # Explosion particles move outward from center
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.velocity_x = math.cos(angle) * speed
            self.velocity_y = math.sin(angle) * speed
    
    def update(self, dt: float) -> None:
        """Update particle state"""
        if not self.is_alive:
            return
        
        self.age += dt
        
        # Check if particle should die
        if self.age >= self.lifetime:
            self.is_alive = False
            return
        
        # Update position
        self.velocity_x += self.acceleration_x * dt
        self.velocity_y += self.acceleration_y * dt
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Update rotation
        self.rotation += self.rotation_speed * dt
        
        # Update visual properties based on age
        progress = self.age / self.lifetime
        
        if self.particle_type == ParticleType.SCORE_POPUP:
            # Fade out over time
            self.alpha = int(self.start_alpha * (1 - progress))
        elif self.particle_type == ParticleType.SPARKLE:
            # Twinkle effect
            twinkle = math.sin(self.age * self.twinkle_speed) * 0.5 + 0.5
            self.alpha = int(self.start_alpha * twinkle)
        elif self.particle_type == ParticleType.EXPLOSION:
            # Fade out and shrink
            self.alpha = int(self.start_alpha * (1 - progress))
            self.size = self.start_size * (1 - progress * 0.5)
        else:
            # Default fade out
            self.alpha = int(self.start_alpha * (1 - progress))
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the particle"""
        if not self.is_alive or self.alpha <= 0:
            return
        
        if self.particle_type == ParticleType.SCORE_POPUP:
            # Draw text particle
            font = pygame.font.Font(None, self.font_size)
            text_surface = font.render(self.text, True, self.color)
            text_surface.set_alpha(self.alpha)
            screen.blit(text_surface, (int(self.x), int(self.y)))
        else:
            # Draw circular particle
            if self.size > 0:
                # Create a surface for the particle with alpha
                particle_surface = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
                color_with_alpha = (*self.color[:3], self.alpha)
                pygame.draw.circle(particle_surface, color_with_alpha, 
                                 (int(self.size), int(self.size)), int(self.size))
                screen.blit(particle_surface, (int(self.x - self.size), int(self.y - self.size)))


class ParticleSystem:
    """Manages collections of particles"""
    
    def __init__(self):
        self.particles: List[Particle] = []
        self.max_particles = 1000
    
    def add_particle(self, particle: Particle) -> None:
        """Add a particle to the system"""
        if len(self.particles) < self.max_particles:
            self.particles.append(particle)
    
    def create_food_consume_effect(self, x: float, y: float, food_color: Tuple[int, int, int]) -> None:
        """Create particles for food consumption"""
        for _ in range(8):
            particle = Particle(
                x, y, ParticleType.FOOD_CONSUME,
                color=food_color,
                size=random.uniform(3, 8),
                lifetime=0.8,
                velocity_x=random.uniform(-3, 3),
                velocity_y=random.uniform(-3, 3)
            )
            self.add_particle(particle)
    
    def create_score_popup(self, x: float, y: float, score: int) -> None:
        """Create a score popup effect"""
        particle = Particle(
            x, y, ParticleType.SCORE_POPUP,
            text=f"+{score}",
            color=SCORE_COLOR,
            lifetime=2.0,
            font_size=FONT_SIZE_LARGE
        )
        self.add_particle(particle)
    
    def create_explosion_effect(self, x: float, y: float, color: Tuple[int, int, int], count: int = 15) -> None:
        """Create an explosion effect"""
        for _ in range(count):
            particle = Particle(
                x, y, ParticleType.EXPLOSION,
                color=color,
                size=random.uniform(2, 6),
                lifetime=1.5
            )
            self.add_particle(particle)
    
    def create_sparkle_effect(self, x: float, y: float, count: int = 5) -> None:
        """Create sparkle effects"""
        for _ in range(count):
            offset_x = random.uniform(-20, 20)
            offset_y = random.uniform(-20, 20)
            particle = Particle(
                x + offset_x, y + offset_y, ParticleType.SPARKLE,
                color=random.choice([WHITE, YELLOW, CYAN]),
                size=random.uniform(1, 3),
                lifetime=2.0,
                twinkle_speed=random.uniform(3, 8)
            )
            self.add_particle(particle)
    
    def update(self, dt: float) -> None:
        """Update all particles"""
        # Update particles
        for particle in self.particles:
            particle.update(dt)
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive]
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw all particles"""
        for particle in self.particles:
            particle.draw(screen)
    
    def clear(self) -> None:
        """Clear all particles"""
        self.particles.clear()


class Animation:
    """Handles smooth animations with easing functions"""
    
    def __init__(self, duration: float, animation_type: AnimationType, 
                 easing_func=EasingFunction.ease_out_quad, **kwargs):
        self.duration = duration
        self.animation_type = animation_type
        self.easing_func = easing_func
        self.elapsed_time = 0.0
        self.is_complete = False
        self.is_active = True
        
        # Animation-specific properties
        self.start_value = kwargs.get('start_value', 0)
        self.end_value = kwargs.get('end_value', 1)
        self.current_value = self.start_value
        
        # For position animations
        self.start_pos = kwargs.get('start_pos', (0, 0))
        self.end_pos = kwargs.get('end_pos', (0, 0))
        self.current_pos = self.start_pos
        
        # For scale animations
        self.start_scale = kwargs.get('start_scale', 1.0)
        self.end_scale = kwargs.get('end_scale', 1.0)
        self.current_scale = self.start_scale
        
        # For color animations
        self.start_color = kwargs.get('start_color', WHITE)
        self.end_color = kwargs.get('end_color', WHITE)
        self.current_color = self.start_color
        
        # Callback function
        self.on_complete = kwargs.get('on_complete', None)
    
    def update(self, dt: float) -> None:
        """Update animation state"""
        if not self.is_active or self.is_complete:
            return
        
        self.elapsed_time += dt
        
        if self.elapsed_time >= self.duration:
            self.elapsed_time = self.duration
            self.is_complete = True
            if self.on_complete:
                self.on_complete()
        
        # Calculate progress (0.0 to 1.0)
        progress = self.elapsed_time / self.duration
        eased_progress = self.easing_func(progress)
        
        # Update values based on animation type
        if self.animation_type in [AnimationType.FADE_IN, AnimationType.FADE_OUT]:
            self.current_value = self.start_value + (self.end_value - self.start_value) * eased_progress
        
        elif self.animation_type in [AnimationType.SCALE_UP, AnimationType.SCALE_DOWN]:
            self.current_scale = self.start_scale + (self.end_scale - self.start_scale) * eased_progress
        
        elif self.animation_type in [AnimationType.SLIDE_IN, AnimationType.SLIDE_OUT]:
            self.current_pos = (
                self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * eased_progress,
                self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * eased_progress
            )
        
        elif self.animation_type == AnimationType.BOUNCE:
            bounce_progress = EasingFunction.bounce_out(progress)
            self.current_value = self.start_value + (self.end_value - self.start_value) * bounce_progress
        
        elif self.animation_type == AnimationType.PULSE:
            # Pulse between start and end values
            pulse_value = math.sin(progress * math.pi * 4) * 0.5 + 0.5
            self.current_value = self.start_value + (self.end_value - self.start_value) * pulse_value
    
    def reset(self) -> None:
        """Reset animation to start"""
        self.elapsed_time = 0.0
        self.is_complete = False
        self.current_value = self.start_value
        self.current_pos = self.start_pos
        self.current_scale = self.start_scale
        self.current_color = self.start_color
    
    def stop(self) -> None:
        """Stop the animation"""
        self.is_active = False


class AnimationManager:
    """Manages multiple animations"""
    
    def __init__(self):
        self.animations: List[Animation] = []
    
    def add_animation(self, animation: Animation) -> None:
        """Add an animation to the manager"""
        self.animations.append(animation)
    
    def create_fade_in(self, duration: float, on_complete=None) -> Animation:
        """Create a fade-in animation"""
        animation = Animation(
            duration, AnimationType.FADE_IN,
            start_value=0, end_value=255,
            on_complete=on_complete
        )
        self.add_animation(animation)
        return animation
    
    def create_fade_out(self, duration: float, on_complete=None) -> Animation:
        """Create a fade-out animation"""
        animation = Animation(
            duration, AnimationType.FADE_OUT,
            start_value=255, end_value=0,
            on_complete=on_complete
        )
        self.add_animation(animation)
        return animation
    
    def create_scale_animation(self, duration: float, start_scale: float, 
                             end_scale: float, on_complete=None) -> Animation:
        """Create a scale animation"""
        animation = Animation(
            duration, AnimationType.SCALE_UP,
            start_scale=start_scale, end_scale=end_scale,
            on_complete=on_complete
        )
        self.add_animation(animation)
        return animation
    
    def create_slide_animation(self, duration: float, start_pos: Tuple[float, float], 
                             end_pos: Tuple[float, float], on_complete=None) -> Animation:
        """Create a slide animation"""
        animation = Animation(
            duration, AnimationType.SLIDE_IN,
            start_pos=start_pos, end_pos=end_pos,
            on_complete=on_complete
        )
        self.add_animation(animation)
        return animation
    
    def update(self, dt: float) -> None:
        """Update all animations"""
        for animation in self.animations:
            animation.update(dt)
        
        # Remove completed animations
        self.animations = [a for a in self.animations if not a.is_complete]
    
    def clear(self) -> None:
        """Clear all animations"""
        self.animations.clear()


class ScreenTransition:
    """Handles screen transitions and effects"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.transition_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.is_transitioning = False
        self.transition_alpha = 0
        self.transition_type = "fade"
        self.transition_duration = 0.5
        self.transition_progress = 0.0
        self.on_transition_complete = None
    
    def start_fade_transition(self, duration: float = 0.5, on_complete=None) -> None:
        """Start a fade transition"""
        self.is_transitioning = True
        self.transition_type = "fade"
        self.transition_duration = duration
        self.transition_progress = 0.0
        self.transition_alpha = 0
        self.on_transition_complete = on_complete
    
    def start_slide_transition(self, direction: str = "left", duration: float = 0.5, on_complete=None) -> None:
        """Start a slide transition"""
        self.is_transitioning = True
        self.transition_type = f"slide_{direction}"
        self.transition_duration = duration
        self.transition_progress = 0.0
        self.on_transition_complete = on_complete
    
    def update(self, dt: float) -> None:
        """Update transition state"""
        if not self.is_transitioning:
            return
        
        self.transition_progress += dt / self.transition_duration
        
        if self.transition_progress >= 1.0:
            self.transition_progress = 1.0
            self.is_transitioning = False
            if self.on_transition_complete:
                self.on_transition_complete()
        
        # Update transition effects
        if self.transition_type == "fade":
            # Fade in/out effect
            if self.transition_progress <= 0.5:
                # Fade to black
                self.transition_alpha = int(255 * (self.transition_progress * 2))
            else:
                # Fade from black
                self.transition_alpha = int(255 * (2 - self.transition_progress * 2))
    
    def draw(self) -> None:
        """Draw transition effects"""
        if not self.is_transitioning:
            return
        
        if self.transition_type == "fade":
            self.transition_surface.fill(BLACK)
            self.transition_surface.set_alpha(self.transition_alpha)
            self.screen.blit(self.transition_surface, (0, 0))


class VisualEffectsManager:
    """Main manager for all visual effects"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.particle_system = ParticleSystem()
        self.animation_manager = AnimationManager()
        self.screen_transition = ScreenTransition(screen)
        
        # Background effects
        self.background_particles: List[Particle] = []
        self.background_effect_timer = 0.0
        self.background_effect_interval = 2.0  # seconds
    
    def create_food_consume_effect(self, x: float, y: float, food_color: Tuple[int, int, int]) -> None:
        """Create visual effect for food consumption"""
        self.particle_system.create_food_consume_effect(x, y, food_color)
        self.particle_system.create_sparkle_effect(x, y, 3)
    
    def create_score_effect(self, x: float, y: float, score: int) -> None:
        """Create visual effect for score increase"""
        self.particle_system.create_score_popup(x, y, score)
    
    def create_game_over_effect(self, x: float, y: float) -> None:
        """Create visual effect for game over"""
        self.particle_system.create_explosion_effect(x, y, RED, 20)
    
    def create_level_up_effect(self) -> None:
        """Create visual effect for level up"""
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        
        # Create multiple sparkle effects
        for i in range(10):
            angle = (i / 10) * 2 * math.pi
            radius = 100
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            self.particle_system.create_sparkle_effect(x, y, 5)
    
    def update_background_effects(self, dt: float) -> None:
        """Update background particle effects"""
        self.background_effect_timer += dt
        
        if self.background_effect_timer >= self.background_effect_interval:
            self.background_effect_timer = 0.0
            
            # Create subtle background sparkles
            x = random.uniform(0, WINDOW_WIDTH)
            y = random.uniform(0, WINDOW_HEIGHT)
            
            particle = Particle(
                x, y, ParticleType.SPARKLE,
                color=random.choice([DARK_GRAY, GRAY]),
                size=random.uniform(1, 2),
                lifetime=3.0,
                twinkle_speed=random.uniform(1, 3),
                velocity_x=random.uniform(-0.5, 0.5),
                velocity_y=random.uniform(-0.5, 0.5)
            )
            self.background_particles.append(particle)
        
        # Update background particles
        for particle in self.background_particles:
            particle.update(dt)
        
        # Remove dead background particles
        self.background_particles = [p for p in self.background_particles if p.is_alive]
    
    def update(self, dt: float) -> None:
        """Update all visual effects"""
        self.particle_system.update(dt)
        self.animation_manager.update(dt)
        self.screen_transition.update(dt)
        self.update_background_effects(dt)
    
    def draw_background_effects(self) -> None:
        """Draw background effects"""
        for particle in self.background_particles:
            particle.draw(self.screen)
    
    def draw_foreground_effects(self) -> None:
        """Draw foreground effects"""
        self.particle_system.draw(self.screen)
        self.screen_transition.draw()
    
    def clear_all_effects(self) -> None:
        """Clear all visual effects"""
        self.particle_system.clear()
        self.animation_manager.clear()
        self.background_particles.clear()


# Utility functions for creating common effects

def create_gradient_surface(width: int, height: int, start_color: Tuple[int, int, int], 
                          end_color: Tuple[int, int, int], vertical: bool = True) -> pygame.Surface:
    """Create a gradient surface"""
    surface = pygame.Surface((width, height))
    
    if vertical:
        for y in range(height):
            ratio = y / height
            color = (
                int(start_color[0] + (end_color[0] - start_color[0]) * ratio),
                int(start_color[1] + (end_color[1] - start_color[1]) * ratio),
                int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
            )
            pygame.draw.line(surface, color, (0, y), (width, y))
    else:
        for x in range(width):
            ratio = x / width
            color = (
                int(start_color[0] + (end_color[0] - start_color[0]) * ratio),
                int(start_color[1] + (end_color[1] - start_color[1]) * ratio),
                int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
            )
            pygame.draw.line(surface, color, (x, 0), (x, height))
    
    return surface


def create_glow_effect(surface: pygame.Surface, color: Tuple[int, int, int], 
                      intensity: int = 3) -> pygame.Surface:
    """Create a glow effect around a surface"""
    glow_surface = pygame.Surface((surface.get_width() + intensity * 2, 
                                  surface.get_height() + intensity * 2), pygame.SRCALPHA)
    
    # Create multiple layers for glow effect
    for i in range(intensity):
        alpha = 255 // (i + 1)
        glow_color = (*color, alpha)
        
        # Draw the surface with offset for glow
        temp_surface = surface.copy()
        temp_surface.fill(glow_color, special_flags=pygame.BLEND_RGBA_MULT)
        
        for x_offset in range(-i, i + 1):
            for y_offset in range(-i, i + 1):
                if x_offset == 0 and y_offset == 0:
                    continue
                glow_surface.blit(temp_surface, (intensity + x_offset, intensity + y_offset))
    
    # Draw the original surface on top
    glow_surface.blit(surface, (intensity, intensity))
    return glow_surface


def interpolate_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], 
                     ratio: float) -> Tuple[int, int, int]:
    """Interpolate between two colors"""
    return (
        int(color1[0] + (color2[0] - color1[0]) * ratio),
        int(color1[1] + (color2[1] - color1[1]) * ratio),
        int(color1[2] + (color2[2] - color1[2]) * ratio)
    )


# Factory functions for common effects

def create_effects_manager(screen: pygame.Surface) -> VisualEffectsManager:
    """Factory function to create a visual effects manager"""
    return VisualEffectsManager(screen)


def create_particle_system() -> ParticleSystem:
    """Factory function to create a particle system"""
    return ParticleSystem()


def create_animation_manager() -> AnimationManager:
    """Factory function to create an animation manager"""
    return AnimationManager()