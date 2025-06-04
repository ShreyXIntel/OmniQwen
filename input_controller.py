"""
Input Controller Module
Handles mouse clicks, keyboard input, and window interactions
"""

import time
import logging
import win32api
import win32con
from config import Config


class InputController:
    def __init__(self):
        self.logger = logging.getLogger(Config.LOGGER_NAME)
    
    def click_at_element(self, element):
        """Click at the center of the given element's bounding box."""
        try:
            # Extract the bounding box
            x1_ratio, y1_ratio, x2_ratio, y2_ratio = element["bbox"]
            
            # Get screen size
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            
            # Calculate center of the element
            center_x = int((x1_ratio + x2_ratio) / 2 * screen_width)
            center_y = int((y1_ratio + y2_ratio) / 2 * screen_height)
            
            return self._perform_click(center_x, center_y, element.get('content', 'Unknown'))
            
        except Exception as e:
            self.logger.error(f"Failed to click at element: {str(e)}")
            return False
    
    def click_at_coordinates(self, x, y, description="coordinates"):
        """Click at specific screen coordinates."""
        return self._perform_click(x, y, description)
    
    def _perform_click(self, x, y, description):
        """Perform the actual mouse click operation."""
        try:
            # Move cursor to position
            win32api.SetCursorPos((x, y))
            time.sleep(Config.CLICK_DELAY)
            
            # Perform mouse click
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
            time.sleep(Config.MOUSE_DOWN_UP_DELAY)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
            
            self.logger.info(f"Clicked on '{description}' at position ({x}, {y})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to perform click at ({x}, {y}): {str(e)}")
            return False
    
    def press_key(self, key_code):
        """Press a keyboard key using the given virtual key code."""
        try:
            # Key down
            win32api.keybd_event(key_code, 0, 0, 0)
            time.sleep(Config.KEY_PRESS_DELAY)
            # Key up
            win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            
            self.logger.info(f"Pressed key with code {key_code}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to press key: {str(e)}")
            return False
    
    def press_escape(self):
        """Press the Escape key."""
        return self.press_key(win32con.VK_ESCAPE)
    
    def press_enter(self):
        """Press the Enter key."""
        return self.press_key(win32con.VK_RETURN)
    
    def press_alt_f4(self):
        """Press Alt+F4 to force close the application."""
        try:
            # Press Alt
            win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
            # Press F4
            win32api.keybd_event(win32con.VK_F4, 0, 0, 0)
            # Release F4
            win32api.keybd_event(win32con.VK_F4, 0, win32con.KEYEVENTF_KEYUP, 0)
            # Release Alt
            win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
            
            self.logger.info("Sent Alt+F4 to force close the application")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Alt+F4: {str(e)}")
            return False
    
    def press_multiple_escape(self, count=5):
        """Press Escape key multiple times in succession."""
        self.logger.info(f"Pressing Escape key {count} times")
        for i in range(count):
            self.press_escape()
            time.sleep(1)
    
    def wait(self, seconds):
        """Wait for a specified number of seconds."""
        self.logger.info(f"Waiting {seconds} seconds")
        time.sleep(seconds)
    
    def get_screen_size(self):
        """Get the current screen dimensions."""
        try:
            width = win32api.GetSystemMetrics(0)
            height = win32api.GetSystemMetrics(1)
            return width, height
        except Exception as e:
            self.logger.error(f"Failed to get screen size: {str(e)}")
            return 1920, 1080  # Default fallback