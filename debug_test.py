"""
DEBUG TEST SCRIPT - Verify that all debug outputs are being created properly.
Run this to test the bulletproof implementations.
"""
import os
import time
import logging
from pathlib import Path

def setup_test_logging():
    """Setup test logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("DebugTest")

def test_debug_outputs():
    """Test that all debug outputs are created."""
    logger = setup_test_logging()
    
    logger.info("=== STARTING DEBUG OUTPUT TEST ===")
    
    try:
        # Import the modules
        import config
        from ui_analyzer import UIAnalyzer
        
        # Create test directories
        test_game = "test_debug_output"
        directories = config.create_run_directory(test_game)
        
        logger.info(f"Created test directories: {directories['root']}")
        
        # Initialize UI Analyzer with forced debug
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
        
        # Take a test screenshot (you can replace this with any image)
        from PIL import ImageGrab
        test_screenshot = ImageGrab.grab()
        screenshot_path = os.path.join(directories["raw_screenshots"], "test_screenshot.png")
        test_screenshot.save(screenshot_path)
        
        logger.info(f"Test screenshot saved: {screenshot_path}")
        
        # Analyze the screenshot - this should create ALL debug outputs
        logger.info("Starting analysis (this should create all debug files)...")
        
        analysis_result = ui_analyzer.analyze_screenshot(
            screenshot_path, directories, "test_context"
        )
        
        logger.info("Analysis completed")
        
        # Check if all debug files were created
        logger.info("=== CHECKING DEBUG FILES ===")
        
        # Check directories
        expected_dirs = [
            "analyzed_screenshots",
            "omniparser_output",
            "qwen_prompt", 
            "qwen_responses",
            "runtime_logs"
        ]
        
        all_good = True
        
        for dir_name in expected_dirs:
            dir_path = directories[dir_name]
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                if files:
                    logger.info(f" {dir_name}: {len(files)} file(s) created")
                    for file in files[:3]:  # Show first 3 files
                        logger.info(f"    - {file}")
                else:
                    logger.error(f" {dir_name}: Directory exists but no files!")
                    all_good = False
            else:
                logger.error(f" {dir_name}: Directory not created!")
                all_good = False
        
        # Check specific files
        logger.info("\n=== FILE CONTENT VERIFICATION ===")
        
        # Check analyzed screenshot
        analyzed_dir = directories["analyzed_screenshots"]
        analyzed_files = [f for f in os.listdir(analyzed_dir) if f.endswith('.png')]
        if analyzed_files:
            logger.info(f" Analyzed screenshot created: {analyzed_files[0]}")
        else:
            logger.error(" No analyzed screenshot found!")
            all_good = False
        
        # Check OmniParser output
        omni_dir = directories["omniparser_output"]
        omni_files = [f for f in os.listdir(omni_dir) if f.endswith('.txt')]
        if omni_files:
            logger.info(f" OmniParser output created: {omni_files[0]}")
            # Show first few lines
            with open(os.path.join(omni_dir, omni_files[0]), 'r') as f:
                lines = f.readlines()[:5]
                logger.info("    Content preview:")
                for line in lines:
                    logger.info(f"    {line.strip()}")
        else:
            logger.error(" No OmniParser output found!")
            all_good = False
        
        # Check Qwen prompt
        prompt_dir = directories["qwen_prompt"]
        prompt_files = [f for f in os.listdir(prompt_dir) if f.endswith('.txt')]
        if prompt_files:
            logger.info(f" Qwen prompt created: {prompt_files[0]}")
        else:
            logger.error(" No Qwen prompt found!")
            all_good = False
        
        # Check Qwen response
        response_dir = directories["qwen_responses"]
        response_files = [f for f in os.listdir(response_dir) if f.endswith('.txt')]
        if response_files:
            logger.info(f" Qwen response created: {response_files[0]}")
        else:
            logger.error(" No Qwen response found!")
            all_good = False
        
        # Check analysis summary
        logs_dir = directories["runtime_logs"]
        summary_files = [f for f in os.listdir(logs_dir) if 'analysis_summary' in f]
        if summary_files:
            logger.info(f" Analysis summary created: {summary_files[0]}")
        else:
            logger.error(" No analysis summary found!")
            all_good = False
        
        logger.info("=== TEST RESULTS ===")
        if all_good:
            logger.info(" ALL DEBUG OUTPUTS CREATED SUCCESSFULLY!")
            logger.info(f"Test results in: {directories['root']}")
        else:
            logger.error(" SOME DEBUG OUTPUTS MISSING!")
            logger.error("Check the error messages above")
        
        # Show analysis result summary
        logger.info(f"\nAnalysis Result Summary:")
        logger.info(f"  - UI Elements: {len(analysis_result['ui_elements'])}")
        logger.info(f"  - Interactive Elements: {len(analysis_result['interactive_elements'])}")
        logger.info(f"  - Decision Action: {analysis_result['decision']['action']}")
        logger.info(f"  - Decision Target: {analysis_result['decision'].get('target', 'None')}")
        logger.info(f"  - Labeled Image: {analysis_result['labeled_image_path']}")
        
        return all_good
        
    except Exception as e:
        logger.error(f" TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print(" Testing debug output creation...")
    print("This will verify that all debug files are created properly.")
    print()
    
    success = test_debug_outputs()
    
    print()
    if success:
        print(" SUCCESS: All debug outputs are working!")
        print("You can now run the benchmarker and see all debug files.")
    else:
        print(" FAILURE: Some debug outputs are missing.")
        print("Check the logs above for details.")