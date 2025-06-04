"""
Main entry point for the Game Benchmarker
Example usage and demonstration of the modularized system
"""

import sys
import argparse
from config import Config
from benchmark_manager import BenchmarkManager


def main():
    """Main function to run the game benchmarker."""
    parser = argparse.ArgumentParser(description="Automated Game Benchmarker")
    parser.add_argument(
        "--game-path",
        type=str,
        default=Config.DEFAULT_GAME_PATH,
        help="Path to the game executable"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=Config.MODEL_PATH,
        help="Path to the OmniParser model"
    )
    parser.add_argument(
        "--test-detection",
        action="store_true",
        help="Run detection test only (no game launch)"
    )
    parser.add_argument(
        "--test-annotations",
        action="store_true",
        help="Run annotation test (takes screenshot and creates debug annotations)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize the benchmark manager
    try:
        benchmarker = BenchmarkManager(
            game_path=args.game_path,
            model_path=args.model_path
        )
        print("✓ Benchmark Manager initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize Benchmark Manager: {e}")
        return 1
    
    # Handle different modes
    if args.status:
        return show_status(benchmarker)
    
    if args.test_detection:
        return test_detection(benchmarker)
    
    if args.test_annotations:
        return test_annotations(benchmarker)
    
    # Run full benchmark cycle
    return run_full_benchmark(benchmarker)


def show_status(benchmarker):
    """Show system status and configuration."""
    print("\n" + "="*50)
    print("GAME BENCHMARKER STATUS")
    print("="*50)
    
    status = benchmarker.get_status_report()
    
    print(f"Game Path: {status['game_path']}")
    print(f"Device: {status['device']}")
    print(f"Model Path: {status['model_path']}")
    print(f"OmniParser Loaded: {'✓' if status['omniparser_loaded'] else '✗'}")
    print(f"Directories Created: {'✓' if status['directories_created'] else '✗'}")
    print(f"Logging Active: {'✓' if status['logging_active'] else '✗'}")
    
    print("\nConfiguration:")
    print(f"  Benchmark Duration: {Config.BENCHMARK_DURATION}s")
    print(f"  Snapshot Interval: {Config.SNAPSHOT_INTERVAL}s")
    print(f"  Box Threshold: {Config.BOX_THRESHOLD}")
    print(f"  Main Menu Wait Time: {Config.MAIN_MENU_WAIT_TIME}s")
    
    print("\nDirectories:")
    for directory in Config.get_directories():
        exists = "✓" if sys.modules['os'].path.exists(directory) else "✗"
        print(f"  {exists} {directory}")
    
    return 0


def test_annotations(benchmarker):
    """Run annotation test with debug visualizations."""
    print("\n" + "="*50)
    print("RUNNING ANNOTATION TEST")
    print("="*50)
    
    try:
        # Take a screenshot of current screen
        snapshot_path = benchmarker.detector.take_snapshot()
        if not snapshot_path:
            print("✗ Failed to take screenshot")
            return 1
        
        print(f"✓ Screenshot taken: {snapshot_path}")
        
        # Parse the image
        parsed_content, labeled_path = benchmarker.detector.parse_image(snapshot_path)
        
        if parsed_content:
            print(f"✓ Detected {len(parsed_content)} UI elements")
            
            # Create debug annotations
            debug_path = benchmarker.detector.create_debug_annotation(snapshot_path, parsed_content)
            
            if debug_path:
                print(f"✓ Debug annotations created: {debug_path}")
                print("✓ Check the omniParsedImages directory for detailed annotations")
                
                # Log some sample elements
                print("\nSample detected elements:")
                for i, element in enumerate(parsed_content[:5]):  # Show first 5
                    if isinstance(element, dict):
                        content = element.get('content', 'N/A')
                        interactive = "interactive" if element.get('interactivity', False) else "non-interactive"
                        print(f"  {i+1}. {content} ({interactive})")
                
                if len(parsed_content) > 5:
                    print(f"  ... and {len(parsed_content) - 5} more elements")
                
                return 0
            else:
                print("✗ Failed to create debug annotations")
                return 1
        else:
            print("⚠ No UI elements detected in the screenshot")
            return 0
            
    except Exception as e:
        print(f"✗ Annotation test failed: {e}")
        return 1


def test_detection(benchmarker):
    """Run detection test."""
    print("\n" + "="*50)
    print("RUNNING DETECTION TEST")
    print("="*50)
    
    success = benchmarker.run_detection_test()
    
    if success:
        print("✓ Detection test completed successfully")
        print("Check the logs for detailed element detection results")
        return 0
    else:
        print("✗ Detection test failed")
        return 1


def run_full_benchmark(benchmarker):
    """Run the full benchmark cycle."""
    print("\n" + "="*50)
    print("STARTING FULL BENCHMARK CYCLE")
    print("="*50)
    
    print("This will:")
    print("1. Launch the game")
    print("2. Wait for main menu")
    print("3. Navigate to benchmark options")
    print("4. Run the benchmark")
    print("5. Exit the game")
    print()
    
    # Confirm before starting
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Benchmark cancelled by user")
        return 0
    
    print("\nStarting benchmark cycle...")
    success = benchmarker.run_full_benchmark_cycle()
    
    if success:
        print("✓ Benchmark cycle completed successfully")
        print("Check the benchmark_logs directory for results and screenshots")
        return 0
    else:
        print("✗ Benchmark cycle failed")
        print("Check the logs for error details")
        return 1


def example_custom_usage():
    """Example of how to use individual components."""
    # This function demonstrates how to use the modularized components
    # individually for more custom control
    
    from detector import VisionDetector
    from input_controller import InputController
    from analyzer import UIAnalyzer
    from game_controller import GameController
    
    # Initialize components individually
    detector = VisionDetector()
    input_ctrl = InputController()
    analyzer = UIAnalyzer(input_ctrl)
    game_ctrl = GameController(r"C:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Far Cry 5\bin\FarCry5.exe")
    
    # Example: Take a screenshot and analyze it
    snapshot_path = detector.take_snapshot()
    if snapshot_path:
        parsed_content, labeled_path = detector.parse_image(snapshot_path)
        
        # Check for specific UI elements
        if analyzer.check_for_main_menu(parsed_content):
            print("Main menu detected!")
        
        # Find and click specific elements
        if analyzer.find_and_click_element(parsed_content, ["play", "start"]):
            print("Clicked on play button")
    
    # Example: Custom keyboard/mouse operations
    input_ctrl.press_escape()
    input_ctrl.wait(2)
    
    # Example: Game-specific operations
    game_ctrl.launch_game()
    main_menu_found, content = game_ctrl.wait_for_main_menu()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)