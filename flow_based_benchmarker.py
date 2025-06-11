"""
Flow-based Game Benchmarker - FIXED VERSION with enhanced debug output and state handling.
"""
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import win32api

from ui_analyzer import UIAnalyzer
from input_controller import InputController
from screenshot_manager import ScreenshotManager

logger = logging.getLogger("FlowBasedGameBenchmarker")

class FlowBasedGameBenchmarker:
    """Fixed flow-based game benchmarker with enhanced debug output and state handling."""
    
    def __init__(self, config_module_path: str = "config", flow_file: str = "game_flows.yaml"):
        """Initialize the flow-based game benchmarker (FIXED VERSION).
        
        Args:
            config_module_path: Path to the configuration module
            flow_file: Path to the YAML flow configuration file
        """
        # Import configuration
        import importlib
        self.config = importlib.import_module(config_module_path)
        
        # FORCE DEBUG MODE ON - FIXED
        self.config.DEBUG_CONFIG["enabled"] = True
        self.config.DEBUG_CONFIG["verbose_logging"] = True
        self.config.DEBUG_CONFIG["save_screenshots"] = True
        self.config.DEBUG_CONFIG["save_omniparser_outputs"] = True
        self.config.DEBUG_CONFIG["save_qwen_responses"] = True
        
        # State tracking
        self.current_game = None
        self.benchmark_started = False
        self.benchmark_completed = False
        self.benchmark_sleep_executed = False
        self.game_process = None  # Always None in NO LAUNCH version
        
        # Don't initialize directories yet - wait for game name
        self.directories = None
        self.logger = None
        
        # Initialize adaptive flow executor with FIXED version
        from adaptive_flow_executor import AdaptiveFlowExecutor
        self.flow_executor = AdaptiveFlowExecutor(flow_file)
        
        # Get screen dimensions
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        
        print("Flow-based GameBenchmarker initialized (FIXED ADAPTIVE STATE MACHINE VERSION)")
        print("DEBUG MODE: FORCED ON")
        print("GAME LAUNCHING: DISABLED")
        print("ADAPTIVE FLOW: ENHANCED")
        print("STATE MACHINE: FIXED")
        print("DEBUG OUTPUT: EVERY ANALYSIS")
    
    def _setup_logging_for_game(self, game_name: str) -> logging.Logger:
        """Setup logging configuration for specific game.
        
        Args:
            game_name: Name of the game for directory creation
            
        Returns:
            Configured logger
        """
        # Create directories for the specific game
        self.directories = self.config.create_run_directory(game_name)
        
        # Setup logging with the new directory
        log_file = os.path.join(self.directories["runtime_logs"], "benchmark.log")
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Setup new logging configuration with DEBUG level
        logging.basicConfig(
            level=logging.DEBUG,  # ALWAYS DEBUG
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # Force reconfiguration
        )
        
        game_logger = logging.getLogger("FlowBasedGameBenchmarker")
        game_logger.info(f"=== FIXED ADAPTIVE STATE MACHINE VERSION LOGGING INITIALIZED ===")
        game_logger.info(f"Game: {game_name}")
        game_logger.info(f"Debug Mode: FORCED ON")
        game_logger.info(f"Game Launch: DISABLED (assumes running)")
        game_logger.info(f"Adaptive Flow: ENHANCED WITH FIXES")
        game_logger.info(f"State Machine: FIXED")
        game_logger.info(f"Debug Output: EVERY ANALYSIS SAVED")
        game_logger.info(f"Confirmation Dialog: IMPROVED DETECTION")
        game_logger.info(f"Log file: {log_file}")
        game_logger.info(f"Directories: {self.directories['root']}")
        game_logger.info(f"Debug config: {self.config.DEBUG_CONFIG}")
        
        return game_logger
    
    def _initialize_components(self):
        """Initialize all the component modules with FIXED versions."""
        try:
            self.logger.info("=== INITIALIZING FIXED COMPONENTS ===")
            
            # Initialize FIXED UI Analyzer with enhanced debug output
            self.ui_analyzer = UIAnalyzer(
                self.config.OMNIPARSER_CONFIG,
                self.config.QWEN_CONFIG,
                self.config.DEBUG_CONFIG  # Debug forced on
            )
            
            # Initialize Input Controller
            self.input_controller = InputController(self.config.INPUT_CONFIG)
            
            # Initialize Screenshot Manager
            self.screenshot_manager = ScreenshotManager(self.directories)
            
            self.logger.info(" All FIXED components initialized successfully")
            self.logger.info(" UI Analyzer: FIXED - Saves debug output for EVERY analysis")
            self.logger.info(" Adaptive Executor: FIXED - Enhanced confirmation dialog detection")
            self.logger.info(" State Machine: FIXED - Better state transitions and loop detection")
            self.logger.info(" Input Controller: win32api")
            self.logger.info(" Screenshot Manager: Ready")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f" Failed to initialize components: {e}")
            else:
                print(f" Failed to initialize components: {e}")
            raise
    
    def run_benchmark_for_game(self, game_name: str, launch_game: bool = False, 
                              game_path: Optional[str] = None) -> bool:
        """Run benchmark for a specific game with FIXED adaptive flow (NO LAUNCH VERSION).
        
        Args:
            game_name: Name of the game (must match YAML flow key)
            launch_game: IGNORED (always False in NO LAUNCH version)
            game_path: IGNORED (not used in NO LAUNCH version)
            
        Returns:
            True if benchmark completed successfully
        """
        try:
            # Set current game first
            self.current_game = game_name
            
            # Setup logging with proper game name
            self.logger = self._setup_logging_for_game(game_name)
            
            self.logger.info("=" * 80)
            self.logger.info(f"  STARTING FIXED ADAPTIVE STATE MACHINE BENCHMARK FOR {game_name.upper()}")
            self.logger.info("  ENHANCED DEBUG OUTPUT - EVERY ANALYSIS SAVED")
            self.logger.info("  IMPROVED CONFIRMATION DIALOG DETECTION")
            self.logger.info("  BETTER STATE TRANSITION HANDLING")
            self.logger.info("=" * 80)
            start_time = time.time()
            
            # Initialize components now that we have directories
            self._initialize_components()
            
            # Set the current game in flow executor
            if not self.flow_executor.set_game(game_name):
                self.logger.error(f" No flow configuration found for game: {game_name}")
                return False
            
            # Set debug directory for flow executor
            self.flow_executor.set_debug_directory(str(self.directories["logs"]))
            
            # Log flow information
            flow_info = self.flow_executor.flows.get(game_name, {})
            benchmark_sleep_time = flow_info.get('benchmark_sleep_time')
            if benchmark_sleep_time:
                self.logger.info(f" Game-specific benchmark sleep time: {benchmark_sleep_time} seconds")
            
            # NO LAUNCH VERSION - Always assume game is running
            self.logger.warning("  NO LAUNCH VERSION: Assuming game is already running!")
            self.logger.warning("  Make sure your game is at the main menu before proceeding!")
            
            # Give game a moment to be ready
            initial_wait = 3
            self.logger.info(f" Waiting {initial_wait} seconds for game to be ready...")
            time.sleep(initial_wait)
            
            # Take initial screenshot to verify game state
            self.logger.info(" Taking initial screenshot to verify game state...")
            initial_screenshot = self.screenshot_manager.take_screenshot(
                custom_name="initial_game_state_check"
            )
            if initial_screenshot:
                self.logger.info(f" Initial screenshot captured: {initial_screenshot}")
                
                # FIXED: Analyze initial screenshot with debug output
                initial_analysis = self.ui_analyzer.analyze_screenshot(
                    initial_screenshot, self.directories, "initial_state_check"
                )
                self.logger.info(f" Initial analysis: {len(initial_analysis['ui_elements'])} elements detected")
                self.logger.info(f" Interactive elements: {len(initial_analysis['interactive_elements'])}")
            else:
                self.logger.error(" Failed to capture initial screenshot!")
                return False
            
            # Execute the complete FIXED flow including enhanced state detection
            if not self._execute_complete_fixed_flow():
                self.logger.error(" Failed to execute complete benchmark flow")
                return False
            
            # Navigate back to main menu after benchmark completes
            if not self._navigate_back_to_main_menu():
                self.logger.warning("  Failed to navigate back to main menu")
            
            # Exit game gracefully from main menu
            if not self._exit_game_gracefully_from_menu():
                self.logger.warning("  Failed to exit game gracefully from menu")
            
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info(f"  FIXED BENCHMARK COMPLETED IN {total_time:.2f} SECONDS")
            self.logger.info("=" * 80)
            
            # Save session summary
            self._save_session_summary(total_time)
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f" Error during benchmark execution: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            else:
                print(f" Error during benchmark execution: {e}")
            return False
        finally:
            self._cleanup_resources()
    
    def _execute_complete_fixed_flow(self) -> bool:
        """Execute the complete FIXED adaptive flow with enhanced state handling.
        
        Returns:
            True if complete flow executed successfully
        """
        try:
            self.logger.info(" EXECUTING FIXED ADAPTIVE FLOW...")
            self.logger.info(" ENHANCEMENTS:")
            self.logger.info("   - Debug output saved for EVERY analysis")
            self.logger.info("   - Improved confirmation dialog detection") 
            self.logger.info("   - Better state transition validation")
            self.logger.info("   - Enhanced loop detection and recovery")
            
            # Set the game in adaptive flow executor
            if not self.flow_executor.set_game(self.current_game):
                self.logger.error(f" Failed to set game in adaptive executor: {self.current_game}")
                return False
            
            # Execute FIXED adaptive flow with enhanced debugging
            success = self.flow_executor.execute_adaptive_flow(
                self.ui_analyzer,
                self.input_controller,
                self.screenshot_manager,
                self.screen_width,
                self.screen_height,
                self.directories
            )
            
            if success:
                self.benchmark_started = True
                self.benchmark_completed = True
                self.benchmark_sleep_executed = True
                self.logger.info(" FIXED ADAPTIVE FLOW COMPLETED SUCCESSFULLY!")
                
                # Try to capture any final results with debug output
                self._attempt_result_capture_with_debug()
                
                return True
            else:
                self.logger.error(" Fixed adaptive flow failed to complete")
                # Log flow summary for debugging
                flow_summary = self.flow_executor.get_flow_summary()
                self.logger.error(f" Flow summary: {flow_summary}")
                return False
                
        except Exception as e:
            self.logger.error(f" Error in fixed adaptive flow execution: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _attempt_result_capture_with_debug(self):
        """Attempt to capture benchmark results with enhanced debug output."""
        try:
            self.logger.info(" ATTEMPTING TO CAPTURE BENCHMARK RESULTS WITH DEBUG...")
            
            max_attempts = 3
            for attempt in range(max_attempts):
                self.logger.info(f" Result capture attempt {attempt + 1}/{max_attempts}")
                
                # Take screenshot for result detection
                screenshot_path = self.screenshot_manager.take_screenshot(
                    custom_name=f"result_attempt_{attempt + 1:02d}"
                )
                if not screenshot_path:
                    continue
                
                # FIXED: Use enhanced result detection with debug output
                result_detection = self.ui_analyzer.detect_benchmark_results(
                    screenshot_path, self.directories
                )
                
                self.logger.info(f" Result detection details: {result_detection}")
                
                if result_detection.get("found", False):
                    self.logger.info(" BENCHMARK RESULTS DETECTED!")
                    
                    # Save the results screenshot to benchmark results folder
                    result_screenshot = self.screenshot_manager.take_benchmark_result_screenshot()
                    if result_screenshot:
                        self.logger.info(f" Results saved: {result_screenshot}")
                    
                    return True
                
                # Wait before next attempt
                if attempt < max_attempts - 1:
                    time.sleep(2)
            
            # Save final screenshot even if no explicit results found
            self.logger.info("  No explicit results detected - saving final screenshot")
            final_screenshot = self.screenshot_manager.take_benchmark_result_screenshot()
            if final_screenshot:
                self.logger.info(f" Final screenshot saved: {final_screenshot}")
            
        except Exception as e:
            self.logger.error(f" Error in result capture: {e}")
    
    def _navigate_back_to_main_menu(self) -> bool:
        """Navigate back to main menu using escape keys and menu detection with debug output."""
        try:
            self.logger.info(" NAVIGATING BACK TO MAIN MENU...")
            
            max_escape_attempts = 8
            
            for attempt in range(max_escape_attempts):
                self.logger.info(f"  Escape attempt {attempt + 1}/{max_escape_attempts}")
                
                # Take screenshot to see current state
                screenshot_path = self.screenshot_manager.take_screenshot(
                    custom_name=f"navigate_back_{attempt + 1:02d}"
                )
                
                if screenshot_path:
                    # FIXED: Analyze current screen with debug output
                    analysis = self.ui_analyzer.analyze_screenshot(
                        screenshot_path, self.directories, "main_menu_check"
                    )
                    
                    # Check for main menu indicators using fuzzy matching
                    main_menu_keywords = ["play", "options", "settings", "exit", "quit", "new game", "continue"]
                    main_menu_elements = self.ui_analyzer.find_elements_by_keywords(
                        analysis["ui_elements"], main_menu_keywords, interactive_only=True
                    )
                    
                    # If we found multiple main menu elements, we're probably at main menu
                    if len(main_menu_elements) >= 2:
                        self.logger.info(f" Main menu detected with {len(main_menu_elements)} menu elements")
                        return True
                
                # Press escape and wait
                self.input_controller.press_key("escape")
                time.sleep(2.5)  # Wait for menu transitions
            
            self.logger.warning("  May have reached main menu area")
            return True  # Return True to continue with exit attempt
            
        except Exception as e:
            self.logger.error(f" Error navigating back to main menu: {e}")
            return False
    
    def _exit_game_gracefully_from_menu(self) -> bool:
        """Exit game gracefully from main menu with debug output."""
        try:
            self.logger.info(" ATTEMPTING GRACEFUL EXIT FROM MAIN MENU...")
            
            max_exit_attempts = 5
            
            for attempt in range(max_exit_attempts):
                self.logger.info(f" Exit attempt {attempt + 1}/{max_exit_attempts}")
                
                # Take screenshot to find exit options
                screenshot_path = self.screenshot_manager.take_screenshot(
                    custom_name=f"exit_attempt_{attempt + 1:02d}"
                )
                
                if not screenshot_path:
                    continue
                
                # FIXED: Analyze for exit options with debug output
                analysis = self.ui_analyzer.analyze_screenshot(
                    screenshot_path, self.directories, "exit_detection"
                )
                
                # Look for exit/quit buttons using fuzzy matching
                exit_keywords = [
                    "exit", "quit", "exit game", "quit game", "exit to desktop",
                    "quit to desktop", "close", "leave game", "return to desktop"
                ]
                
                exit_elements = self.ui_analyzer.find_elements_by_keywords(
                    analysis["ui_elements"], exit_keywords, interactive_only=True
                )
                
                if exit_elements:
                    # Find the best exit element
                    best_exit = self._find_best_exit_element(exit_elements)
                    
                    self.logger.info(f" Found exit element: '{best_exit['content']}'")
                    
                    # Click on the exit element
                    success = self.input_controller.click_element(
                        best_exit, self.screen_width, self.screen_height
                    )
                    
                    if success:
                        self.logger.info(" Exit element clicked, waiting for response...")
                        time.sleep(3)
                        
                        # Check for confirmation dialog with debug output
                        if self._handle_exit_confirmation_with_debug():
                            self.logger.info(" GAME EXITED GRACEFULLY")
                            return True
                    else:
                        self.logger.warning("  Failed to click exit element")
                else:
                    self.logger.warning(f"  No exit elements found in attempt {attempt + 1}")
                
                # Try pressing escape once more before next attempt
                if attempt < max_exit_attempts - 1:
                    self.input_controller.press_key("escape")
                    time.sleep(2)
            
            self.logger.warning("  Graceful exit attempts completed")
            self.logger.warning("  Game may still be running - manual exit may be required")
            return True  # Return True as we've completed our attempts
            
        except Exception as e:
            self.logger.error(f" Error during graceful exit: {e}")
            return False
    
    def _handle_exit_confirmation_with_debug(self) -> bool:
        """Handle exit confirmation dialog with debug output."""
        try:
            # Wait a moment for potential confirmation dialog
            time.sleep(2)
            
            # Take screenshot to check for confirmation
            confirm_screenshot = self.screenshot_manager.take_screenshot(
                custom_name="exit_confirmation_check"
            )
            
            if confirm_screenshot:
                # FIXED: Analyze for confirmation elements with debug output
                analysis = self.ui_analyzer.analyze_screenshot(
                    confirm_screenshot, self.directories, "exit_confirmation"
                )
                
                # Look for confirmation keywords
                confirm_keywords = ["yes", "ok", "confirm", "exit", "quit", "leave"]
                confirm_elements = self.ui_analyzer.find_elements_by_keywords(
                    analysis["ui_elements"], confirm_keywords, interactive_only=True
                )
                
                if confirm_elements:
                    self.logger.info(" Exit confirmation dialog detected")
                    
                    # Click on the first confirmation element
                    success = self.input_controller.click_element(
                        confirm_elements[0], self.screen_width, self.screen_height
                    )
                    
                    if success:
                        self.logger.info(" Confirmation clicked")
                        time.sleep(3)  # Wait for game to close
                        return True
                    else:
                        self.logger.warning("  Failed to click confirmation")
                        return False
                else:
                    self.logger.info("  No confirmation dialog detected")
                    return True  # No confirmation needed
            
            return True
            
        except Exception as e:
            self.logger.error(f" Error handling exit confirmation: {e}")
            return False
    
    def _find_best_exit_element(self, exit_elements: List[Dict]) -> Dict:
        """Find the best exit element from available options."""
        # Priority keywords (higher priority first)
        priority_keywords = [
            "exit to desktop", "quit to desktop", "exit game", "quit game", "exit", "quit"
        ]
        
        for keyword in priority_keywords:
            for element in exit_elements:
                if keyword.lower() in element.get("content", "").lower():
                    return element
        
        # If no priority match, return first element
        return exit_elements[0]
    
    def _save_session_summary(self, total_time: float):
        """Save a comprehensive summary of the benchmarking session with FIXED info."""
        try:
            # Get flow execution summary
            flow_summary = self.flow_executor.get_flow_summary()
            
            session_summary = {
                "timestamp": datetime.now().isoformat(),
                "version": "FIXED_ADAPTIVE_STATE_MACHINE_VERSION",
                "game": self.current_game,
                "total_time_seconds": total_time,
                "benchmark_started": self.benchmark_started,
                "benchmark_sleep_executed": self.benchmark_sleep_executed,
                "benchmark_completed": self.benchmark_completed,
                "screenshots_taken": self.screenshot_manager.get_screenshot_count(),
                "navigation_history": self.input_controller.get_navigation_history(),
                "flow_execution": flow_summary,
                "debug_mode": "FORCED_ON",
                "adaptive_flow": "ENHANCED_FIXED",
                "state_machine": "FIXED",
                "debug_output": "EVERY_ANALYSIS_SAVED",
                "improvements": [
                    "Enhanced confirmation dialog detection",
                    "Debug output saved for every analysis", 
                    "Better state transition validation",
                    "Improved loop detection and recovery",
                    "More precise element clicking"
                ],
                "directories": {key: str(path) for key, path in self.directories.items()}
            }
            
            # Save summary to JSON file
            import json
            summary_path = os.path.join(self.directories["runtime_logs"], "session_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(session_summary, f, indent=2, default=str)
            
            self.logger.info(f" Session summary saved to: {summary_path}")
            
            # Print summary to console
            self.logger.info(" FIXED ADAPTIVE STATE MACHINE BENCHMARK SUMMARY:")
            self.logger.info(f"    Game: {self.current_game}")
            self.logger.info(f"     Total Time: {total_time:.2f}s")
            self.logger.info(f"    Benchmark Started: {' SUCCESS' if self.benchmark_started else ' FAILED'}")
            self.logger.info(f"    Benchmark Sleep Executed: {' SUCCESS' if self.benchmark_sleep_executed else ' FAILED'}")
            self.logger.info(f"    Benchmark Completed: {' SUCCESS' if self.benchmark_completed else ' FAILED'}")
            self.logger.info(f"    Screenshots Taken: {self.screenshot_manager.get_screenshot_count()}")
            self.logger.info(f"    Final State: {flow_summary.get('current_state', 'unknown')}")
            self.logger.info(f"    Total Attempts: {flow_summary.get('total_attempts', 0)}")
            self.logger.info(f"    Goal Reached: {' SUCCESS' if flow_summary.get('goal_reached', False) else ' FAILED'}")
            self.logger.info(f"    Debug Mode: FORCED ON")
            self.logger.info(f"    Adaptive Flow: ENHANCED FIXED")
            self.logger.info(f"    Debug Output: EVERY ANALYSIS SAVED")
            self.logger.info(f"    Results Directory: {self.directories['root']}")
            
            # Also save screenshot summary
            self.screenshot_manager.save_screenshot_summary()
            
        except Exception as e:
            self.logger.error(f" Failed to save session summary: {e}")
    
    def _cleanup_resources(self):
        """Clean up resources."""
        try:
            if self.logger:
                self.logger.info(" Cleaning up resources...")
            
            # NO game process to terminate in NO LAUNCH version
            # self.game_process is always None
            
            # Clean up AI models
            if hasattr(self, 'ui_analyzer'):
                self.ui_analyzer.cleanup()
            
            if self.logger:
                self.logger.info(" Resource cleanup completed")
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"  Error during cleanup: {e}")
            else:
                print(f"  Error during cleanup: {e}")
    
    def get_available_games(self) -> List[str]:
        """Get list of available games from flow configuration."""
        return list(self.flow_executor.flows.keys())
    
    def get_game_flow_info(self, game_name: str) -> Optional[Dict]:
        """Get flow information for a specific game."""
        return self.flow_executor.flows.get(game_name)
    
    def validate_game_flow(self, game_name: str) -> bool:
        """Validate that a game flow is properly configured."""
        game_info = self.get_game_flow_info(game_name)
        if not game_info:
            return False
        
        flow = game_info.get('flow', [])
        if not flow:
            return False
        
        # Check that each step has required fields
        required_fields = ['step', 'screen_context', 'target_button', 'action', 'max_retries']
        for step in flow:
            if not all(field in step for field in required_fields):
                return False
        
        return True