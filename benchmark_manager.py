"""
Benchmark Manager Module
Main orchestration class for the game benchmarking system
"""

import os
import time
import logging
import datetime
from config import Config
from detector import VisionDetector
from input_controller import InputController
from analyzer import UIAnalyzer
from game_controller import GameController


class BenchmarkManager:
    def __init__(self, game_path=None, model_path=None):
        # Setup logging first
        self.setup_logging()
        
        # Create required directories
        self.create_directories()
        
        # Initialize components
        self.game_controller = GameController(game_path)
        self.detector = self.game_controller.detector  # Reuse the detector from game controller
        self.input_controller = self.game_controller.input_controller
        self.analyzer = self.game_controller.analyzer
        
        self.logger = logging.getLogger(Config.LOGGER_NAME)
        self.logger.info("BenchmarkManager initialized successfully")
    
    def setup_logging(self):
        """Set up logging configuration."""
        # Create log directory
        os.makedirs(Config.RUNTIME_LOG_DIR, exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(Config.RUNTIME_LOG_DIR, f"benchmark_run_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format=Config.LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
        )
        
        logger = logging.getLogger(Config.LOGGER_NAME)
        logger.info("Logging initialized")
    
    def create_directories(self):
        """Create necessary directories for operation."""
        directories = Config.get_directories()
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger = logging.getLogger(Config.LOGGER_NAME)
        logger.info("Required directories created/verified")
    
    def run_benchmark(self):
        """Run the in-game benchmark and wait for it to complete."""
        self.logger.info("Starting in-game benchmark...")
        
        # Assume benchmark starts automatically after clicking "benchmark" option in menu
        self.logger.info("Assuming benchmark has started automatically")
        
        # Wait for the specified benchmark time
        self.logger.info(f"Waiting {Config.BENCHMARK_DURATION} seconds for benchmark to complete...")
        self.input_controller.wait(Config.BENCHMARK_DURATION)
        
        # After waiting, check for benchmark results
        self.logger.info("Benchmark time elapsed, checking for results...")
        
        # Take multiple snapshots to ensure we capture the results
        for check in range(Config.MAX_BENCHMARK_RESULT_CHECKS):
            # Take snapshot to check for benchmark results
            snapshot_path = self.detector.take_snapshot()
            if snapshot_path:
                # Parse the snapshot
                parsed_content, labeled_path = self.detector.parse_image(snapshot_path)
                
                # Log detected elements
                self.analyzer.log_all_detected_elements(parsed_content, "Benchmark results")
                
                # Check if we can see FPS metrics, indicating benchmark results
                if self.analyzer.check_for_benchmark_results(parsed_content):
                    self.logger.info("Benchmark results detected!")
                    
                    # Take a special snapshot for the benchmark results
                    result_filepath = self.detector.take_benchmark_result_snapshot()
                    
                    # Try to find any button to acknowledge results
                    if self.analyzer.find_continue_button(parsed_content):
                        self.logger.info("Clicked button to acknowledge benchmark results")
                    
                    return True
            
            # If we didn't find results yet, wait a bit and try again
            if check < Config.MAX_BENCHMARK_RESULT_CHECKS - 1:
                self.logger.info(
                    f"Benchmark results not found yet, checking again in 5 seconds "
                    f"(attempt {check+1}/{Config.MAX_BENCHMARK_RESULT_CHECKS})"
                )
                self.input_controller.wait(5)
        
        # If we still haven't found results, capture the screen anyway
        self.logger.warning("No explicit benchmark results found after completion")
        self.detector.take_completion_snapshot()
        
        self.logger.info("Benchmark phase completed")
        return True
    
    def run_full_benchmark_cycle(self):
        """Run the complete benchmark cycle from launch to exit."""
        try:
            self.logger.info("Starting full benchmark cycle")
            
            # Step 1: Launch the game
            if not self.game_controller.launch_game():
                self.logger.error("Failed to launch game, aborting benchmark cycle")
                return False
            
            # Step 2: Wait for the game to load to main menu
            main_menu_reached, parsed_content = self.game_controller.wait_for_main_menu()
            if not main_menu_reached:
                self.logger.error("Failed to reach main menu, aborting benchmark cycle")
                self.game_controller.exit_game()  # Try to clean up anyway
                return False
            
            # Step 3: Navigate to in-game benchmarks
            if not self.game_controller.navigate_to_benchmarks(parsed_content):
                self.logger.error("Failed to navigate to benchmarks, attempting to exit game")
                self.game_controller.exit_game()
                return False
            
            # Step 4: Run the benchmark
            benchmark_success = self.run_benchmark()
            if not benchmark_success:
                self.logger.warning("Benchmark may not have completed successfully")
                # Continue to exit game even if benchmark didn't succeed perfectly
            
            # Allow some time for the game to return to menu after benchmark completes
            self.logger.info("Waiting for game to settle after benchmark completion")
            self.input_controller.wait(5)
            
            # Step 5: Exit the game gracefully
            self.game_controller.exit_game()
            
            self.logger.info("Benchmark cycle completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during benchmark cycle: {str(e)}")
            # Try to exit the game if an error occurs
            try:
                self.game_controller.exit_game()
            except Exception as exit_error:
                self.logger.error(f"Additionally failed to exit game: {str(exit_error)}")
            return False
    
    def run_detection_test(self):
        """Run a simple test to check if the vision detection is working."""
        self.logger.info("Running detection test...")
        
        try:
            # Take a snapshot
            snapshot_path = self.detector.take_snapshot()
            if not snapshot_path:
                self.logger.error("Failed to take snapshot")
                return False
            
            # Parse the image
            parsed_content, labeled_path = self.detector.parse_image(snapshot_path)
            
            # Log results
            if parsed_content:
                self.logger.info(f"Detection test successful: {len(parsed_content)} elements detected")
                self.analyzer.log_all_detected_elements(parsed_content, "Detection test")
            else:
                self.logger.warning("Detection test completed but no elements detected")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Detection test failed: {str(e)}")
            return False
    
    def get_status_report(self):
        """Get a status report of the system components."""
        status = {
            "game_path": self.game_controller.game_path,
            "device": Config.DEVICE,
            "model_path": Config.MODEL_PATH,
            "directories_created": True,
            "logging_active": True,
        }
        
        try:
            # Test if OmniParser is working
            status["omniparser_loaded"] = (
                self.detector.som_model is not None and 
                self.detector.caption_model_processor is not None
            )
        except:
            status["omniparser_loaded"] = False
        
        return status