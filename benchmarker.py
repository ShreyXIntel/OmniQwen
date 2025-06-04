"""
Main Game Benchmarker orchestrator using OmniParser V2 + Qwen 2.5 7B Instruct.
"""
import os
import logging
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any
import win32api

from ui_analyzer import UIAnalyzer
from input_controller import InputController
from screenshot_manager import ScreenshotManager

logger = logging.getLogger("GameBenchmarker")

class GameBenchmarker:
    """Main orchestrator for automated game benchmarking."""
    
    def __init__(self, config_module_path: str = "config"):
        """Initialize the game benchmarker.
        
        Args:
            config_module_path: Path to the configuration module
        """
        # Import configuration
        import importlib
        self.config = importlib.import_module(config_module_path)
        
        # Initialize directories
        self.directories = self.config.create_run_directory()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # State tracking
        self.benchmark_started = False
        self.benchmark_completed = False
        self.current_attempt = 0
        self.game_process = None
        
        # Get screen dimensions
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        
        self.logger.info("GameBenchmarker initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_file = os.path.join(self.directories["logs"], "benchmark.log")
        logging.basicConfig(
            level=logging.DEBUG if self.config.DEBUG_CONFIG["verbose_logging"] else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("GameBenchmarker")
    
    def _initialize_components(self):
        """Initialize all the component modules."""
        try:
            # Initialize UI Analyzer (OmniParser + Qwen)
            self.ui_analyzer = UIAnalyzer(
                self.config.OMNIPARSER_CONFIG,
                self.config.QWEN_CONFIG,
                self.config.DEBUG_CONFIG
            )
            
            # Initialize Input Controller
            self.input_controller = InputController(self.config.INPUT_CONFIG)
            
            # Initialize Screenshot Manager
            self.screenshot_manager = ScreenshotManager(self.directories)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def launch_game(self, game_name: Optional[str] = None, game_path: Optional[str] = None) -> bool:
        """Launch a game executable.
        
        Args:
            game_name: Name of the game from config
            game_path: Direct path to game executable
            
        Returns:
            True if game launched successfully, False otherwise
        """
        try:
            if game_path:
                launch_path = game_path
            elif game_name:
                launch_path = self.config.get_game_path(game_name)
                if not launch_path:
                    self.logger.error(f"Game path not found for: {game_name}")
                    return False
            else:
                self.logger.info("No game path provided, assuming game is already running")
                return True
            
            self.logger.info(f"Launching game: {launch_path}")
            self.game_process = subprocess.Popen(launch_path)
            
            # Wait for game to start
            initial_wait = self.config.BENCHMARK_CONFIG["initial_wait_time"]
            self.logger.info(f"Waiting {initial_wait} seconds for game to initialize...")
            time.sleep(initial_wait)
            
            # Check if process is still running
            if self.game_process.poll() is not None:
                self.logger.error("Game process terminated unexpectedly")
                return False
            
            self.logger.info("Game launched successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to launch game: {e}")
            return False
    
    def run_benchmark_flow(self) -> bool:
        """Run the complete benchmark flow."""
        try:
            self.logger.info("=== Starting Automated Game Benchmark Flow ===")
            start_time = time.time()
            
            # Step 1: Initial screenshot and analysis
            if not self._analyze_initial_state():
                self.logger.error("Failed to analyze initial game state")
                return False
            
            # Step 2: Navigate to benchmark
            if not self._navigate_to_benchmark():
                self.logger.error("Failed to navigate to benchmark")
                return False
            
            # Step 3: Run benchmark
            if not self._run_benchmark():
                self.logger.error("Failed to run benchmark")
                return False
            
            # Step 4: Capture results
            if not self._capture_benchmark_results():
                self.logger.warning("Failed to capture benchmark results properly")
            
            # Step 5: Navigate back to main menu
            if not self._navigate_back_to_main_menu():
                self.logger.warning("Failed to navigate back to main menu")
            
            # Step 6: Exit game gracefully
            if not self._exit_game_gracefully():
                self.logger.warning("Failed to exit game gracefully")
            
            total_time = time.time() - start_time
            self.logger.info(f"=== Benchmark Flow Completed in {total_time:.2f} seconds ===")
            
            # Save session summary
            self._save_session_summary(total_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during benchmark flow: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            self._cleanup_resources()
    
    def _analyze_initial_state(self) -> bool:
        """Analyze the initial game state."""
        try:
            self.logger.info("Analyzing initial game state...")
            
            # Take initial screenshot
            screenshot_path = self.screenshot_manager.take_screenshot()
            if not screenshot_path:
                return False
            
            # Analyze the screenshot
            analysis = self.ui_analyzer.analyze_screenshot(
                screenshot_path, self.directories, "main_menu"
            )
            
            # Log analysis results
            self.logger.info(f"Initial analysis found {len(analysis['ui_elements'])} UI elements")
            self.logger.info(f"Interactive elements: {len(analysis['interactive_elements'])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error analyzing initial state: {e}")
            return False
    
    def _navigate_to_benchmark(self) -> bool:
        """Navigate through menus to find and start the benchmark."""
        try:
            self.logger.info("Navigating to benchmark...")
            max_attempts = self.config.BENCHMARK_CONFIG["max_navigation_attempts"]
            
            for attempt in range(max_attempts):
                self.logger.info(f"Navigation attempt {attempt + 1}/{max_attempts}")
                
                # Take screenshot
                screenshot_path = self.screenshot_manager.take_screenshot()
                if not screenshot_path:
                    continue
                
                # Check for benchmark option first
                benchmark_detection = self.ui_analyzer.detect_benchmark_option(
                    screenshot_path, self.directories
                )
                
                if benchmark_detection["found"] and benchmark_detection["element"]:
                    self.logger.info("Benchmark option found! Attempting to start...")
                    
                    # Click on benchmark option
                    success = self.input_controller.click_element(
                        benchmark_detection["element"], 
                        self.screen_width, 
                        self.screen_height
                    )
                    
                    if success:
                        self.benchmark_started = True
                        self.logger.info("Benchmark started successfully")
                        return True
                    else:
                        self.logger.warning("Failed to click benchmark option")
                
                # If no benchmark found, analyze for navigation
                analysis = self.ui_analyzer.analyze_screenshot(
                    screenshot_path, self.directories, "navigation"
                )
                
                # Execute navigation decision
                if not self._execute_navigation_decision(analysis):
                    self.logger.warning(f"Navigation failed on attempt {attempt + 1}")
                
                # Check for navigation loops
                if self.input_controller.detect_navigation_loop():
                    self.logger.warning("Navigation loop detected, attempting to break...")
                    self.input_controller.break_navigation_loop()
                
                # Wait before next attempt
                time.sleep(self.config.BENCHMARK_CONFIG["screenshot_interval"])
            
            self.logger.error("Failed to find benchmark after maximum attempts")
            return False
            
        except Exception as e:
            self.logger.error(f"Error navigating to benchmark: {e}")
            return False
    
    def _execute_navigation_decision(self, analysis: Dict) -> bool:
        """Execute a navigation decision from UI analysis."""
        try:
            decision = analysis["decision"]
            target_coordinates = analysis["target_coordinates"]
            
            if decision["confidence"] < self.config.BENCHMARK_CONFIG["confidence_threshold"]:
                self.logger.warning(f"Low confidence decision: {decision['confidence']:.2f}")
            
            return self.input_controller.execute_decision(decision, target_coordinates)
            
        except Exception as e:
            self.logger.error(f"Error executing navigation decision: {e}")
            return False
    
    def _run_benchmark(self) -> bool:
        """Run the benchmark and wait for completion."""
        try:
            self.logger.info("Running benchmark...")
            
            # Wait for benchmark to complete
            benchmark_duration = self.config.BENCHMARK_CONFIG["benchmark_duration"]
            self.logger.info(f"Waiting {benchmark_duration} seconds for benchmark to complete...")
            
            # Take periodic screenshots during benchmark
            screenshots_during_benchmark = []
            interval = 10  # Take screenshot every 10 seconds during benchmark
            
            for i in range(0, benchmark_duration, interval):
                remaining_time = benchmark_duration - i
                wait_time = min(interval, remaining_time)
                
                time.sleep(wait_time)
                
                # Take a screenshot to monitor progress
                screenshot_path = self.screenshot_manager.take_screenshot(
                    custom_name=f"benchmark_progress_{i//interval + 1:02d}"
                )
                if screenshot_path:
                    screenshots_during_benchmark.append(screenshot_path)
                
                self.logger.info(f"Benchmark progress: {i + wait_time}/{benchmark_duration} seconds")
            
            self.logger.info("Benchmark duration completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during benchmark execution: {e}")
            return False
    
    def _capture_benchmark_results(self) -> bool:
        """Capture and save benchmark results."""
        try:
            self.logger.info("Capturing benchmark results...")
            
            max_attempts = self.config.BENCHMARK_CONFIG["result_check_attempts"]
            check_interval = self.config.BENCHMARK_CONFIG["result_check_interval"]
            
            for attempt in range(max_attempts):
                self.logger.info(f"Result capture attempt {attempt + 1}/{max_attempts}")
                
                # Take screenshot
                screenshot_path = self.screenshot_manager.take_screenshot()
                if not screenshot_path:
                    continue
                
                # Check for benchmark results
                result_detection = self.ui_analyzer.detect_benchmark_results(
                    screenshot_path, self.directories
                )
                
                if result_detection["found"]:
                    self.logger.info("Benchmark results detected!")
                    
                    # Save the results screenshot
                    result_screenshot = self.screenshot_manager.take_benchmark_result_screenshot()
                    if result_screenshot:
                        self.logger.info(f"Benchmark results saved: {result_screenshot}")
                    
                    # Log detected metrics
                    if result_detection["metrics"]:
                        self.logger.info(f"Detected metrics: {result_detection['metrics']}")
                    
                    self.benchmark_completed = True
                    return True
                
                # Wait before next attempt
                if attempt < max_attempts - 1:
                    self.logger.info(f"Results not found, waiting {check_interval} seconds...")
                    time.sleep(check_interval)
            
            # If no explicit results found, save final screenshot anyway
            self.logger.warning("No explicit results detected, saving final screenshot")
            result_screenshot = self.screenshot_manager.take_benchmark_result_screenshot()
            if result_screenshot:
                self.logger.info(f"Final screenshot saved: {result_screenshot}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error capturing benchmark results: {e}")
            return False
    
    def _navigate_back_to_main_menu(self) -> bool:
        """Navigate back to the main menu after benchmark completion."""
        try:
            self.logger.info("Navigating back to main menu...")
            
            # Try pressing Escape multiple times to get back to main menu
            for i in range(5):
                self.input_controller.press_key("escape")
                time.sleep(2)
                
                # Take screenshot to check current state
                screenshot_path = self.screenshot_manager.take_screenshot()
                if screenshot_path:
                    analysis = self.ui_analyzer.analyze_screenshot(
                        screenshot_path, self.directories, "main_menu"
                    )
                    
                    # Check if we have main menu elements
                    main_menu_keywords = ["play", "options", "settings", "exit", "quit"]
                    main_menu_elements = self.ui_analyzer.find_elements_by_keywords(
                        analysis["ui_elements"], main_menu_keywords, interactive_only=True
                    )
                    
                    if len(main_menu_elements) >= 2:
                        self.logger.info("Successfully returned to main menu")
                        return True
            
            self.logger.warning("May not have successfully returned to main menu")
            return False
            
        except Exception as e:
            self.logger.error(f"Error navigating back to main menu: {e}")
            return False
    
    def _exit_game_gracefully(self) -> bool:
        """Exit the game gracefully using UI elements."""
        try:
            self.logger.info("Exiting game gracefully...")
            
            # Take screenshot to find exit options
            screenshot_path = self.screenshot_manager.take_screenshot()
            if not screenshot_path:
                return self._force_exit_game()
            
            analysis = self.ui_analyzer.analyze_screenshot(
                screenshot_path, self.directories, "main_menu"
            )
            
            # Look for exit/quit buttons
            exit_keywords = ["exit", "quit", "exit game", "quit game", "exit to desktop"]
            exit_elements = self.ui_analyzer.find_elements_by_keywords(
                analysis["ui_elements"], exit_keywords, interactive_only=True
            )
            
            if exit_elements:
                # Click on the first exit element found
                exit_element = exit_elements[0]
                self.logger.info(f"Clicking on exit element: {exit_element['content']}")
                
                success = self.input_controller.click_element(
                    exit_element, self.screen_width, self.screen_height
                )
                
                if success:
                    # Wait for potential confirmation dialog
                    time.sleep(2)
                    
                    # Check for confirmation dialog
                    confirm_screenshot = self.screenshot_manager.take_screenshot()
                    if confirm_screenshot:
                        confirm_analysis = self.ui_analyzer.analyze_screenshot(
                            confirm_screenshot, self.directories, "confirmation"
                        )
                        
                        # Look for confirmation buttons
                        confirm_keywords = ["yes", "ok", "confirm", "exit"]
                        confirm_elements = self.ui_analyzer.find_elements_by_keywords(
                            confirm_analysis["ui_elements"], confirm_keywords, interactive_only=True
                        )
                        
                        if confirm_elements:
                            self.logger.info("Clicking confirmation button")
                            self.input_controller.click_element(
                                confirm_elements[0], self.screen_width, self.screen_height
                            )
                    
                    self.logger.info("Game exit initiated")
                    time.sleep(3)  # Wait for game to close
                    return True
            
            # If no exit button found, try Alt+F4
            return self._force_exit_game()
            
        except Exception as e:
            self.logger.error(f"Error during graceful exit: {e}")
            return self._force_exit_game()
    
    def _force_exit_game(self) -> bool:
        """Force exit the game using Alt+F4."""
        try:
            self.logger.warning("Force exiting game with Alt+F4...")
            return self.input_controller.emergency_exit()
            
        except Exception as e:
            self.logger.error(f"Error during force exit: {e}")
            return False
    
    def _save_session_summary(self, total_time: float):
        """Save a summary of the benchmarking session."""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "benchmark_started": self.benchmark_started,
                "benchmark_completed": self.benchmark_completed,
                "navigation_attempts": self.current_attempt,
                "screenshots_taken": self.screenshot_manager.get_screenshot_count(),
                "navigation_history": self.input_controller.get_navigation_history(),
                "directories": {key: str(path) for key, path in self.directories.items()}
            }
            
            # Save summary to JSON file
            import json
            summary_path = os.path.join(self.directories["logs"], "session_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Session summary saved to: {summary_path}")
            
            # Also save screenshot summary
            self.screenshot_manager.save_screenshot_summary()
            
        except Exception as e:
            self.logger.error(f"Failed to save session summary: {e}")
    
    def _cleanup_resources(self):
        """Clean up resources and terminate processes."""
        try:
            self.logger.info("Cleaning up resources...")
            
            # Terminate game process if we started it
            if self.game_process and self.game_process.poll() is None:
                self.logger.info("Terminating game process")
                self.game_process.terminate()
                time.sleep(2)
                if self.game_process.poll() is None:
                    self.game_process.kill()
            
            # Clean up AI models
            if hasattr(self, 'ui_analyzer'):
                self.ui_analyzer.cleanup()
            
            # Clean up old screenshots if enabled
            if self.config.DEBUG_CONFIG.get("cleanup_old_screenshots", False):
                deleted_count = self.screenshot_manager.cleanup_old_screenshots(
                    max_age_hours=24, keep_recent_count=20
                )
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old screenshots")
            
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the benchmarker."""
        return {
            "benchmark_started": self.benchmark_started,
            "benchmark_completed": self.benchmark_completed,
            "current_attempt": self.current_attempt,
            "screenshots_taken": self.screenshot_manager.get_screenshot_count(),
            "last_screenshot_time": self.screenshot_manager.get_last_screenshot_time(),
            "navigation_history_length": len(self.input_controller.get_navigation_history())
        }