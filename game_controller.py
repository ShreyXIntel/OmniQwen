"""
Game Controller Module
Handles game launching, menu navigation, and game-specific operations
"""

import time
import logging
import subprocess
import win32con
from config import Config
from detector import VisionDetector
from input_controller import InputController
from analyzer import UIAnalyzer


class GameController:
    def __init__(self, game_path=None):
        self.logger = logging.getLogger(Config.LOGGER_NAME)
        self.game_path = game_path or Config.DEFAULT_GAME_PATH
        
        # Initialize components
        self.detector = VisionDetector()
        self.input_controller = InputController()
        self.analyzer = UIAnalyzer(self.input_controller)
        
        self.logger.info(f"Game path set to: {self.game_path}")
    
    def launch_game(self):
        """Launch the game executable."""
        try:
            self.logger.info(f"Launching game: {self.game_path}")
            subprocess.Popen(self.game_path)
            self.logger.info("Game process started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to launch game: {str(e)}")
            return False
    
    def wait_for_main_menu(self, max_wait_time=None):
        """Wait for the game to load to the main menu."""
        max_wait_time = max_wait_time or Config.MAIN_MENU_WAIT_TIME
        self.logger.info("Waiting for game to load to main menu...")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Take a snapshot if it's time
            if self.detector.should_take_snapshot():
                snapshot_path = self.detector.take_snapshot()
                if snapshot_path:
                    # Parse the snapshot
                    parsed_content, labeled_path = self.detector.parse_image(snapshot_path)
                    
                    # Log detected elements for debugging
                    self.analyzer.log_all_detected_elements(parsed_content)
                    
                    # Check if we're at the main menu
                    if self.analyzer.check_for_main_menu(parsed_content):
                        self.logger.info("Main menu detected!")
                        return True, parsed_content
                    
                    # Log potential main menu elements as fallback
                    for indicators in Config.MAIN_MENU_INDICATORS:
                        if self.analyzer.has_element_containing(parsed_content, indicators):
                            self.logger.info(
                                f"Potential main menu element detected: {indicators}"
                            )
            
            # Short delay before next check
            time.sleep(5)
        
        self.logger.error(
            f"Timed out waiting for main menu after {max_wait_time} seconds"
        )
        return False, []
    
    def navigate_through_menus(self, max_attempts=None):
        """Navigate through menus by trying Escape key and checking for menu options."""
        max_attempts = max_attempts or Config.MAX_MENU_NAVIGATION_ATTEMPTS
        self.logger.info("Attempting to navigate through menus to find exit option...")
        
        for attempt in range(max_attempts):
            # Press Escape to navigate back through menus
            self.input_controller.press_escape()
            self.logger.info(f"Pressed Escape key (attempt {attempt+1}/{max_attempts})")
            
            # Wait for menu transition
            self.input_controller.wait(Config.SHORT_WAIT)
            
            # Take a snapshot and check for menu options
            snapshot_path = self.detector.take_snapshot()
            if snapshot_path:
                parsed_content, _ = self.detector.parse_image(snapshot_path)
                
                # Log all detected menu elements for debugging
                self.analyzer.log_all_detected_elements(parsed_content)
                
                # Check if we're at main menu
                if self.analyzer.check_for_main_menu(parsed_content):
                    self.logger.info("Main menu detected!")
                    return True, parsed_content
                
                # Check for any navigation buttons
                if self.analyzer.find_back_button(parsed_content):
                    self.logger.info("Clicked on back/return button")
                    self.input_controller.wait(Config.SHORT_WAIT)
                    continue
                
                # Check for exit option
                if self.analyzer.find_exit_option(parsed_content):
                    self.logger.info("Found and clicked exit option")
                    return self._handle_exit_confirmation()
            
            # Wait before next attempt
            self.input_controller.wait(1)
        
        self.logger.warning("Failed to navigate through menus after multiple attempts")
        return False, None
    
    def navigate_to_benchmarks(self, parsed_content):
        """Navigate from main menu to in-game benchmarks."""
        self.logger.info("Attempting to navigate to benchmarks...")
        
        # Step 1: Find and click on options/settings
        if not self.analyzer.find_options_menu(parsed_content):
            self.logger.error("Failed to find options/settings menu")
            return False
        
        # Wait for options menu to load
        self.input_controller.wait(Config.MENU_TRANSITION_DELAY)
        
        # Step 2: Look for benchmark option
        benchmark_found = False
        max_attempts = Config.MAX_BENCHMARK_OPTIONS_ATTEMPTS
        
        for attempt in range(max_attempts):
            snapshot_path = self.detector.take_snapshot()
            if snapshot_path:
                parsed_content, _ = self.detector.parse_image(snapshot_path)
                
                # Log elements for debugging
                self.analyzer.log_all_detected_elements(parsed_content, "Options menu")
                
                # Try to find and click on benchmark option
                if self.analyzer.find_benchmark_option(parsed_content):
                    benchmark_found = True
                    break
                
                # If not found, try looking for graphics/display/video options
                if attempt == 1:  # Try graphics menu on second attempt
                    self.analyzer.find_graphics_option(parsed_content)
            
            self.input_controller.wait(Config.MENU_TRANSITION_DELAY)
        
        if not benchmark_found:
            self.logger.error("Failed to find benchmark option after multiple attempts")
            return False
        
        self.logger.info("Successfully navigated to benchmarks")
        return True
    
    def _handle_exit_confirmation(self):
        """Handle exit confirmation dialog if it appears."""
        # Check for confirmation dialog
        self.input_controller.wait(Config.SHORT_WAIT)
        confirm_snapshot = self.detector.take_snapshot()
        
        if confirm_snapshot:
            confirm_content, _ = self.detector.parse_image(confirm_snapshot)
            
            # Look for confirmation buttons
            if self.analyzer.find_confirmation_button(confirm_content):
                self.logger.info("Clicked on confirmation button")
                self.input_controller.wait(Config.CONFIRMATION_WAIT)
                return True, None
        
        return True, None
    
    def exit_game(self):
        """Exit the game gracefully."""
        self.logger.info("Attempting to exit game gracefully...")
        
        # First try to navigate back to main menu
        menu_found, parsed_content = self.navigate_through_menus()
        
        if menu_found and parsed_content:
            # Try to find and click exit option
            if self.analyzer.find_exit_option(parsed_content):
                self.logger.info("Clicked on exit option from main menu")
                return self._handle_exit_confirmation()
        
        # If graceful exit failed, try more aggressive approach
        self.logger.warning("First exit attempt failed, trying more aggressive approach")
        
        # Press ESC multiple times in succession
        self.input_controller.press_multiple_escape(5)
        
        # Try one more time to find an exit button
        snapshot_path = self.detector.take_snapshot()
        if snapshot_path:
            parsed_content, _ = self.detector.parse_image(snapshot_path)
            if self.analyzer.find_exit_option(parsed_content):
                self.logger.info("Found and clicked exit option after multiple ESC presses")
                self.input_controller.wait(Config.MENU_TRANSITION_DELAY)
                return self._handle_exit_confirmation()
        
        # If all else fails, use Alt+F4
        self.logger.warning("Could not exit game gracefully via UI, attempting force close with Alt+F4")
        
        if self.input_controller.press_alt_f4():
            self.input_controller.wait(Config.CONFIRMATION_WAIT)
            
            # Check if we need to confirm exit
            confirm_snapshot = self.detector.take_snapshot()
            if confirm_snapshot:
                confirm_content, _ = self.detector.parse_image(confirm_snapshot)
                if confirm_content:
                    self.analyzer.find_confirmation_button(confirm_content)
            
            return True
        
        return False