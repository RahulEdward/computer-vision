"""OS Controller for system automation"""

import asyncio
import platform
import time
from typing import Optional, List, Tuple
import pyautogui
import pynput
from pynput import mouse, keyboard
import psutil

from computer_genie.utils import setup_logger
from computer_genie.exceptions import PlatformError

logger = setup_logger(__name__)

class OSController:
    """Cross-platform OS controller"""
    
    def __init__(self, display: int = 1):
        self.display = display
        self.platform = platform.system().lower()
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1
        
        # Initialize controllers
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        
        logger.info(f"Initialized OS Controller for {self.platform}")
    
    async def click(
        self, 
        x: int, 
        y: int, 
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.0
    ) -> None:
        """Click at coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate  
            button: Mouse button (left, right, middle)
            clicks: Number of clicks
            interval: Interval between clicks
        """
        try:
            # Move to position
            self.mouse_controller.position = (x, y)
            await asyncio.sleep(0.05)
            
            # Map button
            button_map = {
                "left": mouse.Button.left,
                "right": mouse.Button.right,
                "middle": mouse.Button.middle
            }
            btn = button_map.get(button, mouse.Button.left)
            
            # Perform clicks
            for _ in range(clicks):
                self.mouse_controller.press(btn)
                self.mouse_controller.release(btn)
                if interval > 0:
                    await asyncio.sleep(interval)
            
            logger.debug(f"Clicked at ({x}, {y}) with {button} button")
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            raise
    
    async def double_click(self, x: int, y: int) -> None:
        """Double click at coordinates"""
        await self.click(x, y, clicks=2, interval=0.05)
    
    async def right_click(self, x: int, y: int) -> None:
        """Right click at coordinates"""
        await self.click(x, y, button="right")
    
    async def mouse_move(
        self, 
        x: int, 
        y: int,
        duration: float = 0.0
    ) -> None:
        """Move mouse to coordinates
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration for smooth motion
        """
        if duration > 0:
            # Smooth movement
            start_x, start_y = self.mouse_controller.position
            steps = int(duration * 60)  # 60 FPS
            
            for i in range(steps + 1):
                progress = i / steps
                current_x = int(start_x + (x - start_x) * progress)
                current_y = int(start_y + (y - start_y) * progress)
                self.mouse_controller.position = (current_x, current_y)
                await asyncio.sleep(duration / steps)
        else:
            # Instant movement
            self.mouse_controller.position = (x, y)
        
        logger.debug(f"Moved mouse to ({x}, {y})")
    
    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: str = "left",
        duration: float = 0.5
    ) -> None:
        """Drag from start to end position
        
        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            button: Mouse button to hold
            duration: Drag duration
        """
        # Move to start
        await self.mouse_move(start_x, start_y)
        
        # Press button
        button_map = {
            "left": mouse.Button.left,
            "right": mouse.Button.right,
            "middle": mouse.Button.middle
        }
        btn = button_map.get(button, mouse.Button.left)
        self.mouse_controller.press(btn)
        
        # Drag to end
        await self.mouse_move(end_x, end_y, duration=duration)
        
        # Release button
        self.mouse_controller.release(btn)
        
        logger.debug(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
    
    async def scroll(
        self,
        x: int,
        y: int,
        clicks: int,
        direction: str = "down"
    ) -> None:
        """Scroll at position
        
        Args:
            x: X coordinate
            y: Y coordinate
            clicks: Number of scroll clicks
            direction: Scroll direction (up/down)
        """
        # Move to position
        await self.mouse_move(x, y)
        
        # Scroll
        scroll_amount = clicks if direction == "down" else -clicks
        self.mouse_controller.scroll(0, scroll_amount)
        
        logger.debug(f"Scrolled {direction} by {clicks} at ({x}, {y})")
    
    async def type_text(
        self,
        text: str,
        interval: float = 0.0
    ) -> None:
        """Type text
        
        Args:
            text: Text to type
            interval: Interval between keystrokes
        """
        for char in text:
            self.keyboard_controller.type(char)
            if interval > 0:
                await asyncio.sleep(interval)
        
        logger.debug(f"Typed text: {text[:50]}...")
    
    async def keyboard_tap(
        self,
        key: str,
        modifier_keys: Optional[List[str]] = None
    ) -> None:
        """Press keyboard key with optional modifiers
        
        Args:
            key: Key to press
            modifier_keys: List of modifier keys (ctrl, shift, alt, cmd)
        """
        modifier_keys = modifier_keys or []
        
        # Map keys
        key_map = {
            "enter": keyboard.Key.enter,
            "return": keyboard.Key.enter,
            "tab": keyboard.Key.tab,
            "space": keyboard.Key.space,
            "backspace": keyboard.Key.backspace,
            "delete": keyboard.Key.delete,
            "escape": keyboard.Key.esc,
            "esc": keyboard.Key.esc,
            "up": keyboard.Key.up,
            "down": keyboard.Key.down,
            "left": keyboard.Key.left,
            "right": keyboard.Key.right,
            "home": keyboard.Key.home,
            "end": keyboard.Key.end,
            "pageup": keyboard.Key.page_up,
            "pagedown": keyboard.Key.page_down,
            "f1": keyboard.Key.f1,
            "f2": keyboard.Key.f2,
            "f3": keyboard.Key.f3,
            "f4": keyboard.Key.f4,
            "f5": keyboard.Key.f5,
            "f6": keyboard.Key.f6,
            "f7": keyboard.Key.f7,
            "f8": keyboard.Key.f8,
            "f9": keyboard.Key.f9,
            "f10": keyboard.Key.f10,
            "f11": keyboard.Key.f11,
            "f12": keyboard.Key.f12,
        }
        
        modifier_map = {
            "ctrl": keyboard.Key.ctrl,
            "control": keyboard.Key.ctrl,
            "shift": keyboard.Key.shift,
            "alt": keyboard.Key.alt,
            "cmd": keyboard.Key.cmd,
            "command": keyboard.Key.cmd,
            "win": keyboard.Key.cmd if self.platform == "darwin" else keyboard.Key.cmd,
        }
        
        # Press modifiers
        pressed_modifiers = []
        for mod in modifier_keys:
            if mod.lower() in modifier_map:
                mod_key = modifier_map[mod.lower()]
                self.keyboard_controller.press(mod_key)
                pressed_modifiers.append(mod_key)
        
        # Press main key
        if key.lower() in key_map:
            main_key = key_map[key.lower()]
        else:
            main_key = key
        
        self.keyboard_controller.press(main_key)
        self.keyboard_controller.release(main_key)
        
        # Release modifiers
        for mod_key in reversed(pressed_modifiers):
            self.keyboard_controller.release(mod_key)
        
        logger.debug(f"Pressed {key} with modifiers {modifier_keys}")
    
    async def hotkey(self, *keys) -> None:
        """Press hotkey combination
        
        Args:
            keys: Keys to press together (e.g., 'ctrl', 'c')
        """
        if len(keys) > 1:
            await self.keyboard_tap(keys[-1], modifier_keys=list(keys[:-1]))
        elif keys:
            await self.keyboard_tap(keys[0])
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return self.mouse_controller.position
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size"""
        return pyautogui.size()
    
    async def wait(self, seconds: float) -> None:
        """Wait for specified seconds"""
        await asyncio.sleep(seconds)
    
    def get_active_window_title(self) -> str:
        """Get active window title"""
        if self.platform == "windows":
            import win32gui
            return win32gui.GetWindowText(win32gui.GetForegroundWindow())
        elif self.platform == "darwin":
            # macOS implementation
            import subprocess
            script = '''
            tell application "System Events"
                get name of first application process whose frontmost is true
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        else:
            # Linux implementation
            import subprocess
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
    
    def get_running_processes(self) -> List[str]:
        """Get list of running process names"""
        return [p.name() for p in psutil.process_iter(['name'])]
    
    async def close(self):
        """Cleanup resources"""
        pass