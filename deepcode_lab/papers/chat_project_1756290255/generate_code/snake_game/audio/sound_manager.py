"""
Sound Manager for Snake Game
Handles all audio functionality including sound effects and background music.
"""

import pygame
import os
from pathlib import Path
from typing import Dict, Optional, Any
import threading
import time

from ..utils.constants import (
    DEFAULT_MASTER_VOLUME,
    DEFAULT_SFX_VOLUME,
    DEFAULT_MUSIC_VOLUME,
    ASSETS_DIR,
    SOUNDS_DIR
)


class SoundEffect:
    """Represents a single sound effect with volume control and playback management."""
    
    def __init__(self, file_path: str, volume: float = 1.0):
        """
        Initialize a sound effect.
        
        Args:
            file_path: Path to the sound file
            volume: Volume level (0.0 to 1.0)
        """
        self.file_path = file_path
        self.volume = volume
        self.sound: Optional[pygame.mixer.Sound] = None
        self.loaded = False
        
    def load(self) -> bool:
        """
        Load the sound file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.file_path):
                self.sound = pygame.mixer.Sound(self.file_path)
                self.sound.set_volume(self.volume)
                self.loaded = True
                return True
            else:
                print(f"Warning: Sound file not found: {self.file_path}")
                return False
        except pygame.error as e:
            print(f"Error loading sound {self.file_path}: {e}")
            return False
    
    def play(self, loops: int = 0) -> Optional[pygame.mixer.Channel]:
        """
        Play the sound effect.
        
        Args:
            loops: Number of times to loop (-1 for infinite)
            
        Returns:
            The channel the sound is playing on, or None if failed
        """
        if self.loaded and self.sound:
            try:
                return self.sound.play(loops)
            except pygame.error as e:
                print(f"Error playing sound: {e}")
        return None
    
    def set_volume(self, volume: float):
        """Set the volume of this sound effect."""
        self.volume = max(0.0, min(1.0, volume))
        if self.loaded and self.sound:
            self.sound.set_volume(self.volume)


class MusicManager:
    """Manages background music playback and transitions."""
    
    def __init__(self):
        self.current_music: Optional[str] = None
        self.volume = DEFAULT_MUSIC_VOLUME
        self.is_playing = False
        self.fade_duration = 1000  # milliseconds
        
    def load_music(self, file_path: str) -> bool:
        """
        Load a music file.
        
        Args:
            file_path: Path to the music file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(file_path):
                pygame.mixer.music.load(file_path)
                self.current_music = file_path
                return True
            else:
                print(f"Warning: Music file not found: {file_path}")
                return False
        except pygame.error as e:
            print(f"Error loading music {file_path}: {e}")
            return False
    
    def play(self, loops: int = -1, fade_in: bool = True):
        """
        Play the loaded music.
        
        Args:
            loops: Number of times to loop (-1 for infinite)
            fade_in: Whether to fade in the music
        """
        try:
            if fade_in:
                pygame.mixer.music.play(loops, fade_ms=self.fade_duration)
            else:
                pygame.mixer.music.play(loops)
            pygame.mixer.music.set_volume(self.volume)
            self.is_playing = True
        except pygame.error as e:
            print(f"Error playing music: {e}")
    
    def stop(self, fade_out: bool = True):
        """
        Stop the music.
        
        Args:
            fade_out: Whether to fade out the music
        """
        try:
            if fade_out:
                pygame.mixer.music.fadeout(self.fade_duration)
            else:
                pygame.mixer.music.stop()
            self.is_playing = False
        except pygame.error as e:
            print(f"Error stopping music: {e}")
    
    def pause(self):
        """Pause the music."""
        try:
            pygame.mixer.music.pause()
        except pygame.error as e:
            print(f"Error pausing music: {e}")
    
    def unpause(self):
        """Unpause the music."""
        try:
            pygame.mixer.music.unpause()
        except pygame.error as e:
            print(f"Error unpausing music: {e}")
    
    def set_volume(self, volume: float):
        """Set the music volume."""
        self.volume = max(0.0, min(1.0, volume))
        try:
            pygame.mixer.music.set_volume(self.volume)
        except pygame.error as e:
            print(f"Error setting music volume: {e}")


class SoundManager:
    """
    Central sound management system for the Snake Game.
    Handles sound effects, background music, and volume controls.
    """
    
    def __init__(self):
        """Initialize the sound manager."""
        self.initialized = False
        self.master_volume = DEFAULT_MASTER_VOLUME
        self.sfx_volume = DEFAULT_SFX_VOLUME
        self.music_volume = DEFAULT_MUSIC_VOLUME
        
        # Sound effects dictionary
        self.sound_effects: Dict[str, SoundEffect] = {}
        
        # Music manager
        self.music_manager = MusicManager()
        
        # Audio channels for better control
        self.channels = {
            'sfx': None,
            'ui': None,
            'ambient': None
        }
        
        # Initialize pygame mixer
        self._initialize_mixer()
        
        # Load default sounds
        self._load_default_sounds()
    
    def _initialize_mixer(self):
        """Initialize the pygame mixer system."""
        try:
            # Initialize mixer with specific settings for better quality
            pygame.mixer.pre_init(
                frequency=44100,    # Sample rate
                size=-16,          # 16-bit signed samples
                channels=2,        # Stereo
                buffer=1024        # Buffer size
            )
            pygame.mixer.init()
            
            # Set number of mixing channels
            pygame.mixer.set_num_channels(8)
            
            self.initialized = True
            print("Sound system initialized successfully")
            
        except pygame.error as e:
            print(f"Failed to initialize sound system: {e}")
            self.initialized = False
    
    def _load_default_sounds(self):
        """Load default sound effects for the game."""
        # Define default sound files
        default_sounds = {
            'eat_food': 'eat.wav',
            'game_over': 'game_over.wav',
            'level_up': 'level_up.wav',
            'menu_select': 'menu_select.wav',
            'menu_navigate': 'menu_navigate.wav',
            'pause': 'pause.wav',
            'unpause': 'unpause.wav',
            'high_score': 'high_score.wav'
        }
        
        # Create sounds directory path
        sounds_path = Path(ASSETS_DIR) / SOUNDS_DIR
        
        # Load each sound effect
        for sound_name, filename in default_sounds.items():
            file_path = sounds_path / filename
            self.load_sound_effect(sound_name, str(file_path))
    
    def load_sound_effect(self, name: str, file_path: str, volume: float = 1.0) -> bool:
        """
        Load a sound effect.
        
        Args:
            name: Identifier for the sound effect
            file_path: Path to the sound file
            volume: Volume level (0.0 to 1.0)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.initialized:
            return False
        
        sound_effect = SoundEffect(file_path, volume * self.sfx_volume * self.master_volume)
        if sound_effect.load():
            self.sound_effects[name] = sound_effect
            return True
        return False
    
    def play_sound_effect(self, name: str, loops: int = 0) -> bool:
        """
        Play a sound effect.
        
        Args:
            name: Identifier of the sound effect
            loops: Number of times to loop
            
        Returns:
            True if played successfully, False otherwise
        """
        if not self.initialized or name not in self.sound_effects:
            return False
        
        channel = self.sound_effects[name].play(loops)
        return channel is not None
    
    def load_background_music(self, file_path: str) -> bool:
        """
        Load background music.
        
        Args:
            file_path: Path to the music file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.initialized:
            return False
        
        return self.music_manager.load_music(file_path)
    
    def play_background_music(self, loops: int = -1, fade_in: bool = True):
        """
        Play background music.
        
        Args:
            loops: Number of times to loop (-1 for infinite)
            fade_in: Whether to fade in the music
        """
        if self.initialized:
            self.music_manager.set_volume(self.music_volume * self.master_volume)
            self.music_manager.play(loops, fade_in)
    
    def stop_background_music(self, fade_out: bool = True):
        """
        Stop background music.
        
        Args:
            fade_out: Whether to fade out the music
        """
        if self.initialized:
            self.music_manager.stop(fade_out)
    
    def pause_background_music(self):
        """Pause background music."""
        if self.initialized:
            self.music_manager.pause()
    
    def unpause_background_music(self):
        """Unpause background music."""
        if self.initialized:
            self.music_manager.unpause()
    
    def set_master_volume(self, volume: float):
        """
        Set the master volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.master_volume = max(0.0, min(1.0, volume))
        self._update_all_volumes()
    
    def set_sfx_volume(self, volume: float):
        """
        Set the sound effects volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.sfx_volume = max(0.0, min(1.0, volume))
        self._update_sfx_volumes()
    
    def set_music_volume(self, volume: float):
        """
        Set the music volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.music_volume = max(0.0, min(1.0, volume))
        self.music_manager.set_volume(self.music_volume * self.master_volume)
    
    def _update_all_volumes(self):
        """Update all volume levels based on master volume."""
        self._update_sfx_volumes()
        self.music_manager.set_volume(self.music_volume * self.master_volume)
    
    def _update_sfx_volumes(self):
        """Update all sound effect volumes."""
        for sound_effect in self.sound_effects.values():
            current_volume = sound_effect.volume / (self.sfx_volume * self.master_volume) if (self.sfx_volume * self.master_volume) > 0 else 1.0
            sound_effect.set_volume(current_volume * self.sfx_volume * self.master_volume)
    
    def get_volume_info(self) -> Dict[str, float]:
        """
        Get current volume information.
        
        Returns:
            Dictionary with volume levels
        """
        return {
            'master': self.master_volume,
            'sfx': self.sfx_volume,
            'music': self.music_volume
        }
    
    def is_music_playing(self) -> bool:
        """
        Check if music is currently playing.
        
        Returns:
            True if music is playing, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            return pygame.mixer.music.get_busy()
        except pygame.error:
            return False
    
    def stop_all_sounds(self):
        """Stop all currently playing sounds and music."""
        if self.initialized:
            try:
                pygame.mixer.stop()  # Stop all sound effects
                self.music_manager.stop(fade_out=False)  # Stop music
            except pygame.error as e:
                print(f"Error stopping all sounds: {e}")
    
    def cleanup(self):
        """Clean up the sound system."""
        if self.initialized:
            self.stop_all_sounds()
            try:
                pygame.mixer.quit()
            except pygame.error as e:
                print(f"Error during sound cleanup: {e}")
            self.initialized = False


# Global sound manager instance
_sound_manager: Optional[SoundManager] = None


def get_sound_manager() -> SoundManager:
    """
    Get the global sound manager instance.
    
    Returns:
        The global SoundManager instance
    """
    global _sound_manager
    if _sound_manager is None:
        _sound_manager = SoundManager()
    return _sound_manager


def create_sound_manager() -> SoundManager:
    """
    Create a new sound manager instance.
    
    Returns:
        A new SoundManager instance
    """
    return SoundManager()


# Convenience functions for common operations
def play_sound(name: str, loops: int = 0) -> bool:
    """Play a sound effect using the global sound manager."""
    return get_sound_manager().play_sound_effect(name, loops)


def play_music(file_path: str, loops: int = -1, fade_in: bool = True) -> bool:
    """Load and play background music using the global sound manager."""
    manager = get_sound_manager()
    if manager.load_background_music(file_path):
        manager.play_background_music(loops, fade_in)
        return True
    return False


def stop_music(fade_out: bool = True):
    """Stop background music using the global sound manager."""
    get_sound_manager().stop_background_music(fade_out)


def set_volume(master: Optional[float] = None, sfx: Optional[float] = None, music: Optional[float] = None):
    """Set volume levels using the global sound manager."""
    manager = get_sound_manager()
    if master is not None:
        manager.set_master_volume(master)
    if sfx is not None:
        manager.set_sfx_volume(sfx)
    if music is not None:
        manager.set_music_volume(music)