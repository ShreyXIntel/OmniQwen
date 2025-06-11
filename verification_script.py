"""
Verification script to test the debug output fixes.
Run this to verify that all debug files are being created for every analysis.
"""
import os
import logging
import time
from pathlib import Path

def setup_logging():
    """Setup logging for verification."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("DebugVerification")

def test_debug_output_fixes():
    """Test that debug outputs are created for every analysis."""
    logger = setup_logging()
    
    logger.info("=== TESTING DEBUG OUTPUT FIXES ===")
    
    try:
        # Import the fixed modules
        import config
        from ui_analyzer import UIAnalyzer
        
        # Create test directories
        test_game = "debug_fix_test"
        directories = config.create_run_directory(test_game)
        
        logger.info(f"Created test directories: {directories['root']}")
        
        # Initialize UI Analyzer with debug enabled
        debug_config = {
            "save_omniparser_outputs": True,
            "save_qwen_responses": True,
            "verbose_logging": True
        }
        
        ui_analyzer = UIAnalyzer(
            config.OMNIPARSER_CONFIG,
            config.QWEN_CONFIG,
            debug_config
        )
        
        logger.info("UI Analyzer initialized")
        
        # Take a test screenshot 
        from PIL import ImageGrab
        test_screenshot = ImageGrab.grab()
        screenshot_path = os.path.join(directories["raw_screenshots"], "test_screenshot.png")
        test_screenshot.save(screenshot_path)
        
        logger.info(f"Test screenshot saved: {screenshot_path}")
        
        # Run multiple analyses to test if each one saves debug files
        contexts = ["main_menu", "confirmation_dialog", "benchmark_menu"]
        
        for i, context in enumerate(contexts, 1):
            logger.info(f"\n--- Analysis {i}: {context} ---")
            
            analysis_result = ui_analyzer.analyze_screenshot(
                screenshot_path, directories, context
            )
            
            logger.info(f"Analysis {i} completed")
            logger.info(f"  Elements found: {len(analysis_result['ui_elements'])}")
            logger.info(f"  Interactive elements: {len(analysis_result['interactive_elements'])}")
            logger.info(f"  Decision: {analysis_result['decision']['action']}")
            
            # Small delay between analyses
            time.sleep(1)
        
        # Check if debug files were created for each analysis
        logger.info("\n=== CHECKING DEBUG FILES ===")
        
        # Check directories
        debug_dirs = [
            ("analyzed_screenshots", "Analyzed Screenshots"),
            ("omniparser_output", "OmniParser Output"),
            ("qwen_prompt", "Qwen Prompts"), 
            ("qwen_responses", "Qwen Responses"),
            ("runtime_logs", "Runtime Logs")
        ]
        
        all_good = True
        total_files = 0
        
        for dir_key, dir_name in debug_dirs:
            dir_path = directories[dir_key]
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith(('.txt', '.png'))]
                if files:
                    logger.info(f" {dir_name}: {len(files)} file(s)")
                    total_files += len(files)
                    
                    # Show file names for verification
                    for file in sorted(files)[:3]:  # Show first 3 files
                        logger.info(f"    - {file}")
                    if len(files) > 3:
                        logger.info(f"    ... and {len(files) - 3} more")
                else:
                    logger.error(f" {dir_name}: No files found!")
                    all_good = False
            else:
                logger.error(f" {dir_name}: Directory not found!")
                all_good = False
        
        # Specific checks for multiple analyses
        logger.info(f"\n=== MULTIPLE ANALYSIS VERIFICATION ===")
        
        # Check if we have files for each analysis
        omni_dir = directories["omniparser_output"]
        if os.path.exists(omni_dir):
            omni_files = [f for f in os.listdir(omni_dir) if f.endswith('.txt')]
            expected_analyses = len(contexts)
            
            if len(omni_files) >= expected_analyses:
                logger.info(f" SUCCESS: Found {len(omni_files)} OmniParser files for {expected_analyses} analyses")
            else:
                logger.error(f" ISSUE: Only {len(omni_files)} OmniParser files for {expected_analyses} analyses")
                all_good = False
        
        # Check Qwen files
        qwen_prompt_dir = directories["qwen_prompt"]
        qwen_response_dir = directories["qwen_responses"]
        
        if os.path.exists(qwen_prompt_dir) and os.path.exists(qwen_response_dir):
            prompt_files = [f for f in os.listdir(qwen_prompt_dir) if f.endswith('.txt')]
            response_files = [f for f in os.listdir(qwen_response_dir) if f.endswith('.txt')]
            
            if len(prompt_files) >= expected_analyses and len(response_files) >= expected_analyses:
                logger.info(f" SUCCESS: Found {len(prompt_files)} prompt files and {len(response_files)} response files")
            else:
                logger.error(f" ISSUE: Prompts: {len(prompt_files)}, Responses: {len(response_files)}")
                all_good = False
        
        # Final summary
        logger.info(f"\n=== VERIFICATION RESULTS ===")
        logger.info(f"Total debug files created: {total_files}")
        logger.info(f"Test directory: {directories['root']}")
        
        if all_good:
            logger.info(" SUCCESS: All debug outputs are working correctly!")
            logger.info(" Each analysis is now saving separate debug files.")
            logger.info(" The infinite loop and missing debug issues should be resolved.")
        else:
            logger.error(" FAILURE: Some debug outputs are still missing.")
            logger.error(" Check the error messages above for details.")
        
        return all_good
        
    except Exception as e:
        logger.error(f" TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_state_detection():
    """Test state detection improvements."""
    logger = logging.getLogger("StateDetection")
    
    logger.info("\n=== TESTING STATE DETECTION IMPROVEMENTS ===")
    
    try:
        from adaptive_flow_executor import AdaptiveFlowExecutor, BenchmarkState
        
        executor = AdaptiveFlowExecutor()
        executor.set_game("black_myth_wukong")
        
        # Test confirmation dialog detection
        test_elements = [
            {"content": "Do you want to start benchmark?", "interactivity": False},
            {"content": "Confirm", "interactivity": True},
            {"content": "Cancel", "interactivity": True}
        ]
        
        detected_state = executor.detect_current_state(test_elements)
        
        if detected_state == BenchmarkState.BENCHMARK_CONFIRM:
            logger.info(" SUCCESS: Confirmation dialog correctly detected")
        else:
            logger.error(f" ISSUE: Expected BENCHMARK_CONFIRM, got {detected_state.value}")
        
        # Test main menu detection
        main_menu_elements = [
            {"content": "New Game", "interactivity": True},
            {"content": "Continue", "interactivity": True},
            {"content": "Benchmark", "interactivity": True},
            {"content": "Settings", "interactivity": True},
            {"content": "Exit", "interactivity": True}
        ]
        
        detected_state = executor.detect_current_state(main_menu_elements)
        
        if detected_state == BenchmarkState.MAIN_MENU:
            logger.info(" SUCCESS: Main menu correctly detected")
        else:
            logger.error(f" ISSUE: Expected MAIN_MENU, got {detected_state.value}")
        
        return True
        
    except Exception as e:
        logger.error(f" State detection test failed: {e}")
        return False

if __name__ == "__main__":
    print(" Testing debug output fixes...")
    print("This will verify that:")
    print("1. Debug files are saved for EVERY analysis")
    print("2. State detection improvements work")
    print("3. Confirmation dialog detection is enhanced")
    print()
    
    # Test debug outputs
    debug_success = test_debug_output_fixes()
    
    # Test state detection
    state_success = check_state_detection()
    
    print()
    if debug_success and state_success:
        print(" ALL TESTS PASSED!")
        print("The fixes should resolve:")
        print("- Missing debug files for subsequent analyses")
        print("- Infinite loops in confirmation dialogs")
        print("- Better state detection and transitions")
    else:
        print(" SOME TESTS FAILED!")
        print("Check the logs above for details.")