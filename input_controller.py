"""
Input Controller module for handling mouse and keyboard interactions using win32api.
Updated to remove unicode characters and improve error logging.
"""
import logging
import time
from typing import Tuple, Optional, List, Dict, Any
import win32api
import win32con
import win32gui

logger = logging.getLogger("InputController")

class InputController:
    """Handles mouse and keyboard interactions using win32api."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the input controller.
        
        Args:
            config: Configuration dictionary for input settings
        """
        self.config = config
        self.navigation_history = []
        self.last_action_time = 0
        
        logger.info("Input Controller initialized")
    
    def click_at_coordinates(self, x: int, y: int) -> bool:
        """Click at specific screen coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if click was successful, False otherwise
        """
        try:
            logger.info(f"Clicking at coordinates: ({x}, {y})")
            
            # Get foreground window to ensure we're clicking in the right context
            hwnd = win32gui.GetForegroundWindow()
            
            # Move cursor to target position
            if self.config.get("smooth_mouse_movement", True):
                self._move_cursor_smoothly(x, y)
            else:
                win32api.SetCursorPos((x, y))
                time.sleep(0.1)
            
            # Perform left mouse click
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(self.config.get("click_delay", 0.2))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            
            # Record the action
            self._record_action("CLICK", coordinates=(x, y))
            
            # Wait after action
            time.sleep(self.config.get("post_action_delay", 1.0))
            
            logger.info(f"Successfully clicked at ({x}, {y})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to click at coordinates ({x}, {y}): {e}")
            return False
    
    def click_element(self, element: Dict, screen_width: int, screen_height: int) -> bool:
        """Click on a UI element using its bounding box.
        
        Args:
            element: UI element dictionary with bbox
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            True if click was successful, False otherwise
        """
        try:
            bbox = element.get("bbox", [])
            if len(bbox) != 4:
                logger.error("Invalid bounding box in element")
                return False
            
            # Convert normalized coordinates to screen coordinates
            x1_ratio, y1_ratio, x2_ratio, y2_ratio = bbox
            x1 = int(x1_ratio * screen_width)
            y1 = int(y1_ratio * screen_height)
            x2 = int(x2_ratio * screen_width)
            y2 = int(y2_ratio * screen_height)
            
            # Calculate center coordinates
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            element_name = element.get("content", "Unknown")
            logger.info(f"Clicking on element '{element_name}' at center ({center_x}, {center_y})")
            
            return self.click_at_coordinates(center_x, center_y)
            
        except Exception as e:
            logger.error(f"Failed to click element: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """Press a keyboard key or key combination.
        
        Args:
            key: Key to press (e.g., 'escape', 'alt+f4', 'enter')
            
        Returns:
            True if key press was successful, False otherwise
        """
        try:
            key = key.lower().strip()
            logger.info(f"Pressing key: {key}")
            
            if key == "escape":
                self._press_single_key(win32con.VK_ESCAPE)
            elif key == "enter":
                self._press_single_key(win32con.VK_RETURN)
            elif key == "space":
                self._press_single_key(win32con.VK_SPACE)
            elif key == "tab":
                self._press_single_key(win32con.VK_TAB)
            elif key == "f12":
                self._press_single_key(win32con.VK_F12)
            elif key == "alt+f4":
                self._press_key_combination([win32con.VK_MENU, win32con.VK_F4])
            elif key == "ctrl+c":
                self._press_key_combination([win32con.VK_CONTROL, ord('C')])
            elif key == "ctrl+v":
                self._press_key_combination([win32con.VK_CONTROL, ord('V')])
            elif key.startswith("f") and key[1:].isdigit():
                # Function keys F1-F12
                f_num = int(key[1:])
                if 1 <= f_num <= 12:
                    f_key = win32con.VK_F1 + (f_num - 1)
                    self._press_single_key(f_key)
                else:
                    logger.error(f"Invalid function key: {key}")
                    return False
            elif len(key) == 1 and key.isalpha():
                # Single letter key
                self._press_single_key(ord(key.upper()))
            else:
                logger.error(f"Unsupported key: {key}")
                return False
            
            # Record the action
            self._record_action("KEY_PRESS", key=key)
            
            # Wait after key press
            time.sleep(self.config.get("post_action_delay", 1.0))
            
            logger.info(f"Successfully pressed key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to press key '{key}': {e}")
            return False
    
    def execute_decision(self, decision: Dict, target_coordinates: Optional[Tuple[int, int]] = None) -> bool:
        """Execute a decision from the UI analyzer.
        
        Args:
            decision: Decision dictionary from UI analyzer
            target_coordinates: Optional target coordinates for CLICK actions
            
        Returns:
            True if action was executed successfully, False otherwise
        """
        try:
            action = decision.get("action", "").upper()
            confidence = decision.get("confidence", 0.0)
            
            logger.info(f"Executing decision: {action} (confidence: {confidence:.2f})")
            
            if action == "CLICK":
                if target_coordinates:
                    x, y = target_coordinates
                    return self.click_at_coordinates(x, y)
                else:
                    logger.error("CLICK action requested but no target coordinates provided")
                    return False
                    
            elif action == "BACK":
                return self.press_key("escape")
                
            elif action == "WAIT":
                wait_time = 3.0  # Default wait time
                logger.info(f"Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return True
                
            elif action == "EXIT":
                return self.press_key("alt+f4")
                
            else:
                logger.error(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute decision: {e}")
            return False
    
    def _move_cursor_smoothly(self, target_x: int, target_y: int) -> None:
        """Move cursor smoothly to target coordinates.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
        """
        try:
            current_x, current_y = win32api.GetCursorPos()
            steps = self.config.get("mouse_movement_steps", 10)
            
            # Calculate step size
            x_step = (target_x - current_x) / steps
            y_step = (target_y - current_y) / steps
            
            # Move cursor in steps
            for i in range(steps):
                new_x = int(current_x + x_step * (i + 1))
                new_y = int(current_y + y_step * (i + 1))
                win32api.SetCursorPos((new_x, new_y))
                time.sleep(0.01)  # Small delay between steps
            
            # Ensure final position is exact
            win32api.SetCursorPos((target_x, target_y))
            time.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"Smooth cursor movement failed, using direct movement: {e}")
            win32api.SetCursorPos((target_x, target_y))
            time.sleep(0.1)
    
    def _press_single_key(self, vk_code: int) -> None:
        """Press a single key using virtual key code.
        
        Args:
            vk_code: Virtual key code
        """
        # Key down
        win32api.keybd_event(vk_code, 0, 0, 0)
        time.sleep(self.config.get("key_press_delay", 0.1))
        # Key up
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    
    def _press_key_combination(self, vk_codes: List[int]) -> None:
        """Press a combination of keys.
        
        Args:
            vk_codes: List of virtual key codes to press together
        """
        # Press all keys down
        for vk_code in vk_codes:
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.05)
        
        time.sleep(self.config.get("key_press_delay", 0.1))
        
        # Release all keys in reverse order
        for vk_code in reversed(vk_codes):
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(0.05)
    
    def _record_action(self, action_type: str, **kwargs) -> None:
        """Record an action in the navigation history.
        
        Args:
            action_type: Type of action performed
            **kwargs: Additional action parameters
        """
        action_record = {
            "action": action_type,
            "timestamp": time.time(),
            **kwargs
        }
        
        self.navigation_history.append(action_record)
        self.last_action_time = time.time()
        
        # Keep only last 50 actions to prevent memory bloat
        if len(self.navigation_history) > 50:
            self.navigation_history = self.navigation_history[-50:]
    
    def detect_navigation_loop(self, max_history: int = 6, time_threshold: float = 30.0) -> bool:
        """Detect if we're stuck in a navigation loop.
        
        Args:
            max_history: Number of recent actions to check
            time_threshold: Time threshold for detecting loops in seconds
            
        Returns:
            True if navigation loop is detected, False otherwise
        """
        if len(self.navigation_history) < max_history:
            return False
        
        try:
            # Check recent actions within time threshold
            current_time = time.time()
            recent_actions = [
                action for action in self.navigation_history[-max_history:]
                if current_time - action["timestamp"] <= time_threshold
            ]
            
            if len(recent_actions) < max_history:
                return False
            
            # Look for repeating patterns
            pattern_size = max_history // 2
            if pattern_size < 2:
                return False
            
            pattern_1 = [action["action"] for action in recent_actions[:pattern_size]]
            pattern_2 = [action["action"] for action in recent_actions[pattern_size:2*pattern_size]]
            
            if pattern_1 == pattern_2:
                logger.warning(f"Navigation loop detected! Pattern: {pattern_1}")
                return True
            
            # Also check for excessive clicking in the same area
            click_actions = [action for action in recent_actions if action["action"] == "CLICK"]
            if len(click_actions) >= 4:
                # Check if clicks are in similar locations
                coordinates = [action.get("coordinates") for action in click_actions if action.get("coordinates")]
                if len(coordinates) >= 4:
                    # Check if all clicks are within a small area (50x50 pixels)
                    x_coords = [coord[0] for coord in coordinates]
                    y_coords = [coord[1] for coord in coordinates]
                    
                    if (max(x_coords) - min(x_coords) <= 50 and 
                        max(y_coords) - min(y_coords) <= 50):
                        logger.warning("Detected excessive clicking in same area")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting navigation loop: {e}")
            return False
    
    def break_navigation_loop(self) -> bool:
        """Attempt to break out of a navigation loop.
        
        Returns:
            True if break attempt was successful, False otherwise
        """
        try:
            logger.info("Attempting to break navigation loop...")
            
            # Try pressing Escape multiple times
            for i in range(3):
                self.press_key("escape")
                time.sleep(1)
            
            # Clear recent navigation history to reset loop detection
            self.navigation_history = self.navigation_history[:-6] if len(self.navigation_history) > 6 else []
            
            logger.info("Navigation loop break attempt completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to break navigation loop: {e}")
            return False
    
    def get_navigation_history(self) -> List[Dict]:
        """Get the navigation action history.
        
        Returns:
            List of navigation action records
        """
        return self.navigation_history.copy()
    
    def clear_navigation_history(self) -> None:
        """Clear the navigation history."""
        self.navigation_history.clear()
        logger.info("Navigation history cleared")
    
    def wait_for_action_cooldown(self) -> None:
        """Wait for action cooldown period to prevent rapid successive actions."""
        if self.last_action_time > 0:
            time_since_last = time.time() - self.last_action_time
            min_interval = 0.5  # Minimum 500ms between actions
            
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                logger.debug(f"Waiting {wait_time:.2f}s for action cooldown")
                time.sleep(wait_time)
    
    def emergency_exit(self) -> bool:
        """Perform emergency exit using multiple methods.
        
        Returns:
            True if emergency exit was attempted, False otherwise
        """
        try:
            logger.warning("Performing emergency exit...")
            
            # Try Alt+F4 multiple times
            for i in range(3):
                self.press_key("alt+f4")
                time.sleep(2)
                
                # Check if we need confirmation
                time.sleep(1)
            
            # If that doesn't work, try Escape multiple times
            for i in range(5):
                self.press_key("escape")
                time.sleep(1)
            
            logger.info("Emergency exit sequence completed")
            return True
            
        except Exception as e:
            logger.error(f"Emergency exit failed: {e}")
            return False