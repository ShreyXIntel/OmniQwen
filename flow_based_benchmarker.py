"""
Flow-based Game Benchmarker using YAML navigation flows with OmniParser V2 + Qwen 2.5.
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
from flow_executor import FlowExecutor

logger = logging.getLogger("FlowBasedGameBenchmarker")

class FlowBasedGameBenchmarker:
    """Flow-based game benchmarker using YAML configuration for navigation."""
    
    def __init__(self, config_module_path: str = "config", flow_file: str = "game_flows.yaml"):
        """Initialize the flow-based game benchmarker.
        
        Args:
            config_module_path: Path to the configuration module
            flow_file: Path to the YAML flow configuration file
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
        
        # Initialize flow executor
        self.flow_executor = FlowExecutor(flow_file)
        
        # State tracking
        self.current_game = None
        self.benchmark_started = False
        self.benchmark_completed = False
        self.game_process = None
        
        # Get screen dimensions
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        
        self.logger.info("Flow-based GameBenchmarker initialized successfully")
    
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
        return logging.getLogger("FlowBasedGameBenchmarker")
    
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
    
    def run_benchmark_for_game(self, game_name: str, launch_game: bool = False, 
                              game_path: Optional[str] = None) -> bool:
        """Run benchmark for a specific game using its flow configuration.
        
        Args:
            game_name: Name of the game (must match YAML flow key)
            launch_game: Whether to launch the game executable
            game_path: Custom path to game executable
            
        Returns:
            True if benchmark completed successfully
        """
        try:
            self.logger.info(f"=== Starting Flow-Based Benchmark for {game_name} ===")
            start_time = time.time()
            
            # Set the current game in flow executor
            if not self.flow_executor.set_game(game_name):
                self.logger.error(f"No flow configuration found for game: {game_name}")
                return False
            
            self.current_game = game_name
            
            # Launch game if requested
            if launch_game:
                if not self._launch_game(game_name, game_path):
                    self.logger.error("Failed to launch game")
                    return False
            else:
                self.logger.info("Assuming game is already running...")
                # Wait a moment for game to be ready
                time.sleep(3)
            
            # Execute the navigation flow
            if not self._execute_navigation_flow():
                self.logger.error("Failed to navigate to benchmark")
                return False
            
            # Run the benchmark
            if not self._run_benchmark():
                self.logger.error("Failed to run benchmark")
                return False
            
            # Capture results
            if not self._capture_benchmark_results():
                self.logger.warning("Failed to capture benchmark results properly")
            
            # Exit game gracefully
            if not self._exit_game_gracefully():
                self.logger.warning("Failed to exit game gracefully")
            
            total_time = time.time() - start_time
            self.logger.info(f"=== Flow-Based Benchmark Completed in {total_time:.2f} seconds ===")
            
            # Save session summary
            self._save_session_summary(total_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during benchmark execution: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            self._cleanup_resources()
    
    def _launch_game(self, game_name: str, custom_path: Optional[str] = None) -> bool:
        """Launch the game executable.
        
        Args:
            game_name: Name of the game
            custom_path: Custom path to game executable
            
        Returns:
            True if game launched successfully
        """
        try:
            if custom_path:
                launch_path = custom_path
            else:
                launch_path = self.config.get_game_path(game_name)
                if not launch_path:
                    self.logger.error(f"Game path not found for: {game_name}")
                    return False
            
            self.logger.info(f"Launching {game_name}: {launch_path}")
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
    
    def _execute_navigation_flow(self) -> bool:
        """Execute the navigation flow to reach the benchmark.
        
        Returns:
            True if navigation completed successfully
        """
        try:
            self.logger.info("Executing navigation flow to reach benchmark...")
            
            # Initial screenshot to see starting state
            initial_screenshot = self.screenshot_manager.take_screenshot(
                custom_name="flow_start_state"
            )
            
            flow_timeout = self.config.BENCHMARK_CONFIG.get("benchmark_timeout", 120)
            flow_start_time = time.time()
            
            while not self.flow_executor.is_flow_complete():
                # Check overall timeout
                if time.time() - flow_start_time > flow_timeout:
                    self.logger.error(f"Navigation flow timed out after {flow_timeout} seconds")
                    return False
                
                # Get current step info
                step_info = self.flow_executor.get_current_step_info()
                if not step_info:
                    self.logger.error("No current step available")
                    break
                
                # Execute current step with retry logic
                result = self.flow_executor.execute_current_step(
                    self.ui_analyzer,
                    self.input_controller,
                    self.screenshot_manager,
                    self.screen_width,
                    self.screen_height,
                    self.directories
                )
                
                if result.success:
                    self.logger.info(f"✅ Step {step_info.step} completed successfully")
                    
                    # Check if this was the final benchmark trigger step
                    if self._is_benchmark_trigger_step(step_info):
                        self.benchmark_started = True
                        self.logger.info("🚀 Benchmark has been triggered!")
                        break
                    
                    # Advance to next step
                    if not self.flow_executor.advance_step():
                        self.logger.info("Reached end of navigation flow")
                        break
                else:
                    self.logger.error(f"❌ Step {step_info.step} failed: {result.error_message}")
                    self.logger.error(f"Detected elements: {result.detected_elements}")
                    
                    # Save current state for debugging
                    debug_screenshot = self.screenshot_manager.take_screenshot(
                        custom_name=f"step_{step_info.step}_failed"
                    )
                    
                    return False
            
            # Check if benchmark was triggered
            if self.benchmark_started:
                self.logger.info("✅ Navigation flow completed - benchmark started")
                return True
            else:
                self.logger.warning("⚠ Navigation flow completed but benchmark may not have started")
                return True  # Continue anyway, might be auto-started
                
        except Exception as e:
            self.logger.error(f"Error in navigation flow execution: {e}")
            return False
    
    def _is_benchmark_trigger_step(self, step_info) -> bool:
        """Check if the current step is expected to trigger the benchmark.
        
        Args:
            step_info: Current step information
            
        Returns:
            True if this step should trigger benchmark start
        """
        # Check if target button contains benchmark-related terms
        target = step_info.target_button.upper()
        trigger_keywords = ["BENCHMARK", "START", "RUN", "GO", "CONFIRM"]
        
        return any(keyword in target for keyword in trigger_keywords)
    
    def _run_benchmark(self) -> bool:
        """Run the benchmark and wait for completion.
        
        Returns:
            True if benchmark completed successfully
        """
        try:
            self.logger.info("Running benchmark...")
            
            # Get benchmark duration from flow settings or config
            benchmark_duration = (
                self.flow_executor.settings.get('benchmark_duration') or
                self.config.BENCHMARK_CONFIG["benchmark_duration"]
            )
            
            self.logger.info(f"Waiting {benchmark_duration} seconds for benchmark to complete...")
            
            # Take periodic screenshots during benchmark
            progress_interval = 15  # Screenshot every 15 seconds
            screenshots_taken = 0
            
            for elapsed in range(0, benchmark_duration, progress_interval):
                remaining = min(progress_interval, benchmark_duration - elapsed)
                time.sleep(remaining)
                
                # Take progress screenshot
                screenshots_taken += 1
                progress_screenshot = self.screenshot_manager.take_screenshot(
                    custom_name=f"benchmark_progress_{screenshots_taken:02d}"
                )
                
                progress_percent = ((elapsed + remaining) / benchmark_duration) * 100
                self.logger.info(f"Benchmark progress: {progress_percent:.1f}% ({elapsed + remaining}/{benchmark_duration}s)")
                
                # Check if benchmark completed early
                if self.flow_executor.detect_benchmark_results(
                    self.ui_analyzer, self.screenshot_manager
                ):
                    self.logger.info("🎉 Benchmark completed early - results detected!")
                    self.benchmark_completed = True
                    return True
            
            self.logger.info("⏰ Benchmark duration elapsed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during benchmark execution: {e}")
            return False
    
    def _capture_benchmark_results(self) -> bool:
        """Capture and save benchmark results.
        
        Returns:
            True if results captured successfully
        """
        try:
            self.logger.info("Capturing benchmark results...")
            
            max_attempts = self.config.BENCHMARK_CONFIG["result_check_attempts"]
            check_interval = self.config.BENCHMARK_CONFIG["result_check_interval"]
            
            for attempt in range(max_attempts):
                self.logger.info(f"Result capture attempt {attempt + 1}/{max_attempts}")
                
                # Use flow executor's result detection
                results_found = self.flow_executor.detect_benchmark_results(
                    self.ui_analyzer, self.screenshot_manager
                )
                
                if results_found:
                    self.logger.info("📊 Benchmark results detected!")
                    
                    # Save the results screenshot
                    result_screenshot = self.screenshot_manager.take_benchmark_result_screenshot()
                    if result_screenshot:
                        self.logger.info(f"Results saved: {result_screenshot}")
                    
                    self.benchmark_completed = True
                    return True
                
                # Wait before next attempt
                if attempt < max_attempts - 1:
                    self.logger.info(f"Results not found, waiting {check_interval} seconds...")
                    time.sleep(check_interval)
            
            # If no explicit results found, save final screenshot anyway
            self.logger.warning("⚠ No explicit results detected, saving final screenshot")
            final_screenshot = self.screenshot_manager.take_benchmark_result_screenshot()
            if final_screenshot:
                self.logger.info(f"Final screenshot saved: {final_screenshot}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error capturing benchmark results: {e}")
            return False
    
    def _exit_game_gracefully(self) -> bool:
        """Exit the game gracefully using flow configuration.
        
        Returns:
            True if game exited successfully
        """
        try:
            self.logger.info("Exiting game gracefully using flow configuration...")
            
            # Get exit flow steps
            exit_steps = self.flow_executor.get_exit_flow()
            
            if not exit_steps:
                self.logger.warning("No exit flow configured, using emergency exit")
                return self._emergency_exit()
            
            # Execute exit flow
            for step in exit_steps:
                self.logger.info(f"Exit step: {step.description}")
                
                # Create temporary flow executor state for exit
                temp_step_info = step
                
                # Try to execute the exit step
                success = False
                for attempt in range(step.max_retries):
                    try:
                        if step.action == "press_escape":
                            success = self.input_controller.press_key("escape")
                        elif step.action == "click":
                            # Take screenshot and find exit button
                            screenshot_path = self.screenshot_manager.take_screenshot()
                            if screenshot_path:
                                analysis_result = self.ui_analyzer.analyze_screenshot(
                                    screenshot_path, self.directories, "exit_menu"
                                )
                                
                                # Find exit button
                                exit_elements = self.ui_analyzer.find_elements_by_keywords(
                                    analysis_result["ui_elements"], step.expected_buttons, True
                                )
                                
                                if exit_elements:
                                    success = self.input_controller.click_element(
                                        exit_elements[0], self.screen_width, self.screen_height
                                    )
                        
                        if success:
                            time.sleep(2)  # Wait for action to take effect
                            break
                            
                    except Exception as e:
                        self.logger.warning(f"Exit step attempt {attempt + 1} failed: {e}")
                
                if not success:
                    self.logger.warning(f"Exit step failed: {step.description}")
            
            # Final wait for game to close
            time.sleep(3)
            self.logger.info("Game exit sequence completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during graceful exit: {e}")
            return self._emergency_exit()
    
    def _emergency_exit(self) -> bool:
        """Emergency exit using Alt+F4 and other methods.
        
        Returns:
            True if emergency exit attempted
        """
        try:
            self.logger.warning("🚨 Performing emergency exit...")
            return self.input_controller.emergency_exit()
        except Exception as e:
            self.logger.error(f"Emergency exit failed: {e}")
            return False
    
    def _save_session_summary(self, total_time: float):
        """Save a comprehensive summary of the benchmarking session.
        
        Args:
            total_time: Total execution time in seconds
        """
        try:
            # Get flow execution summary
            flow_summary = self.flow_executor.get_flow_summary()
            
            session_summary = {
                "timestamp": datetime.now().isoformat(),
                "game": self.current_game,
                "total_time_seconds": total_time,
                "benchmark_started": self.benchmark_started,
                "benchmark_completed": self.benchmark_completed,
                "screenshots_taken": self.screenshot_manager.get_screenshot_count(),
                "navigation_history": self.input_controller.get_navigation_history(),
                "flow_execution": flow_summary,
                "directories": {key: str(path) for key, path in self.directories.items()}
            }
            
            # Save summary to JSON file
            import json
            summary_path = os.path.join(self.directories["logs"], "session_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(session_summary, f, indent=2, default=str)
            
            self.logger.info(f"Session summary saved to: {summary_path}")
            
            # Print summary to console
            self.logger.info("📋 BENCHMARK SUMMARY:")
            self.logger.info(f"   Game: {self.current_game}")
            self.logger.info(f"   Total Time: {total_time:.2f}s")
            self.logger.info(f"   Benchmark Started: {'✅' if self.benchmark_started else '❌'}")
            self.logger.info(f"   Benchmark Completed: {'✅' if self.benchmark_completed else '❌'}")
            self.logger.info(f"   Screenshots Taken: {self.screenshot_manager.get_screenshot_count()}")
            self.logger.info(f"   Flow Steps Completed: {flow_summary.get('completed_steps', 0)}/{flow_summary.get('total_steps', 0)}")
            
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
            
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def get_available_games(self) -> List[str]:
        """Get list of available games from flow configuration.
        
        Returns:
            List of available game names
        """
        return list(self.flow_executor.flows.keys())
    
    def get_game_flow_info(self, game_name: str) -> Optional[Dict]:
        """Get flow information for a specific game.
        
        Args:
            game_name: Name of the game
            
        Returns:
            Game flow information or None if not found
        """
        return self.flow_executor.flows.get(game_name)
    
    def validate_game_flow(self, game_name: str) -> bool:
        """Validate that a game flow is properly configured.
        
        Args:
            game_name: Name of the game to validate
            
        Returns:
            True if flow is valid
        """
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