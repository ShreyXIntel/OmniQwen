"""
Detection diagnostic script to analyze why some elements aren't detected.
Run this on your current screenshot to see exactly what's being detected.
"""
import os
import logging
from PIL import Image

def diagnose_detection(screenshot_path="latest_screenshot.png"):
    """Diagnose detection issues with current screenshot."""
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("DetectionDiagnostic")
    
    try:
        # Import modules
        import config
        from ui_analyzer import UIAnalyzer
        
        # Test with CURRENT settings
        logger.info("=== TESTING WITH CURRENT SETTINGS ===")
        
        ui_analyzer = UIAnalyzer(
            config.OMNIPARSER_CONFIG,
            config.QWEN_CONFIG, 
            {"save_omniparser_outputs": True, "save_qwen_responses": True}
        )
        
        # Create test directory
        test_dirs = config.create_run_directory("detection_test")
        
        # Analyze the screenshot
        result = ui_analyzer.analyze_screenshot(
            screenshot_path, test_dirs, "diagnostic"
        )
        
        logger.info(f"CURRENT SETTINGS RESULTS:")
        logger.info(f"  Total elements detected: {len(result['ui_elements'])}")
        logger.info(f"  Interactive elements: {len(result['interactive_elements'])}")
        
        # List all detected elements
        logger.info("=== ALL DETECTED ELEMENTS ===")
        for i, element in enumerate(result['ui_elements']):
            content = element.get('content', '').strip()
            interactive = element.get('interactivity', False)
            bbox = element.get('bbox', [])
            logger.info(f"{i+1:2d}. '{content}' ({'Interactive' if interactive else 'Static'}) {bbox}")
        
        # Check specifically for benchmark-related terms
        logger.info("=== BENCHMARK-RELATED DETECTION ===")
        benchmark_keywords = ['benchmark', 'test', 'performance']
        benchmark_elements = []
        
        for element in result['ui_elements']:
            content = element.get('content', '').lower()
            for keyword in benchmark_keywords:
                if keyword in content:
                    benchmark_elements.append(element)
                    logger.info(f"FOUND: '{element.get('content')}' (contains '{keyword}')")
        
        if not benchmark_elements:
            logger.warning(" NO BENCHMARK-RELATED ELEMENTS DETECTED!")
            logger.warning("This explains why the flow can't find the benchmark option.")
        
        # Test with LOWERED settings
        logger.info("\n=== TESTING WITH LOWERED THRESHOLDS ===")
        
        # Use aggressive settings
        aggressive_config = {
            "icon_detect_model_path": "weights/icon_detect/model.pt",
            "caption_model_name": "florence2",
            "caption_model_path": "weights/icon_caption_florence", 
            "device": "cuda",
            "box_threshold": 0.01,  # VERY LOW
            "iou_threshold": 0.3,   # VERY LOW
            "use_local_semantics": True,
            "batch_size": 128,
            "scale_img": False,
            "easyocr_args": {
                "paragraph": False,
                "text_threshold": 0.5  # VERY LOW
            },
            "use_paddleocr": True
        }
        
        ui_analyzer_aggressive = UIAnalyzer(
            aggressive_config,
            config.QWEN_CONFIG,
            {"save_omniparser_outputs": True, "save_qwen_responses": True}
        )
        
        result_aggressive = ui_analyzer_aggressive.analyze_screenshot(
            screenshot_path, test_dirs, "diagnostic_aggressive"
        )
        
        logger.info(f"AGGRESSIVE SETTINGS RESULTS:")
        logger.info(f"  Total elements detected: {len(result_aggressive['ui_elements'])}")
        logger.info(f"  Interactive elements: {len(result_aggressive['interactive_elements'])}")
        
        # Check for benchmark elements again
        benchmark_elements_aggressive = []
        for element in result_aggressive['ui_elements']:
            content = element.get('content', '').lower()
            for keyword in benchmark_keywords:
                if keyword in content:
                    benchmark_elements_aggressive.append(element)
                    logger.info(f"FOUND (AGGRESSIVE): '{element.get('content')}' (contains '{keyword}')")
        
        # Compare results
        logger.info(f"\n=== COMPARISON ===")
        logger.info(f"Current settings: {len(result['ui_elements'])} elements")
        logger.info(f"Aggressive settings: {len(result_aggressive['ui_elements'])} elements")
        logger.info(f"Improvement: +{len(result_aggressive['ui_elements']) - len(result['ui_elements'])} elements")
        
        if len(benchmark_elements_aggressive) > len(benchmark_elements):
            logger.info(" AGGRESSIVE SETTINGS FOUND MORE BENCHMARK ELEMENTS!")
            logger.info("Recommendation: Use the lowered thresholds in config.py")
        
        # Save results to file
        summary_file = os.path.join(test_dirs["runtime_logs"], "detection_diagnostic.txt")
        with open(summary_file, 'w') as f:
            f.write("=== DETECTION DIAGNOSTIC RESULTS ===\n")
            f.write(f"Screenshot: {screenshot_path}\n")
            f.write(f"Current settings: {len(result['ui_elements'])} elements\n")
            f.write(f"Aggressive settings: {len(result_aggressive['ui_elements'])} elements\n")
            f.write(f"Benchmark elements found (current): {len(benchmark_elements)}\n")
            f.write(f"Benchmark elements found (aggressive): {len(benchmark_elements_aggressive)}\n")
            
            f.write(f"\n=== ALL ELEMENTS (CURRENT) ===\n")
            for i, element in enumerate(result['ui_elements']):
                content = element.get('content', '').strip()
                interactive = element.get('interactivity', False)
                f.write(f"{i+1:2d}. '{content}' ({'Interactive' if interactive else 'Static'})\n")
            
            f.write(f"\n=== ALL ELEMENTS (AGGRESSIVE) ===\n")
            for i, element in enumerate(result_aggressive['ui_elements']):
                content = element.get('content', '').strip()
                interactive = element.get('interactivity', False)
                f.write(f"{i+1:2d}. '{content}' ({'Interactive' if interactive else 'Static'})\n")
        
        logger.info(f"Detailed results saved to: {summary_file}")
        logger.info(f"Check analyzed screenshots in: {test_dirs['analyzed_screenshots']}")
        
        return len(result_aggressive['ui_elements']) > len(result['ui_elements'])
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print(" Running detection diagnostic...")
    print("This will test current vs. lowered detection thresholds")
    
    # You can specify your screenshot path here
    screenshot_path = input("Enter screenshot path (or press Enter for latest): ").strip()
    if not screenshot_path:
        # Try to find latest screenshot
        if os.path.exists("benchmark_runs"):
            import glob
            latest_runs = glob.glob("benchmark_runs/*/Raw_Screenshots/*.png")
            if latest_runs:
                screenshot_path = max(latest_runs, key=os.path.getctime)
                print(f"Using latest screenshot: {screenshot_path}")
    
    if not screenshot_path or not os.path.exists(screenshot_path):
        print(" No screenshot found. Please provide a valid path.")
        exit(1)
    
    success = diagnose_detection(screenshot_path)
    
    if success:
        print(" AGGRESSIVE SETTINGS DETECTED MORE ELEMENTS!")
        print(" Recommendation: Update your config.py with the lowered thresholds")
    else:
        print("  No significant improvement with aggressive settings")
        print(" The issue might be elsewhere in the pipeline")