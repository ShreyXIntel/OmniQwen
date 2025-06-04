"""
Debug verification script to test OmniParser annotations and Qwen logging.
Run this to verify the fixes are working.
"""
import os
import sys
import logging
from PIL import ImageGrab
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DebugVerification")

def test_omniparser_annotations():
    """Test OmniParser annotation generation."""
    try:
        from omniparser import OmniParserV2
        import config
        
        logger.info("Testing OmniParser annotation generation...")
        
        # Initialize OmniParser
        omniparser = OmniParserV2(config.OMNIPARSER_CONFIG)
        
        # Take a test screenshot
        screenshot = ImageGrab.grab()
        test_screenshot_path = "test_screenshot.png"
        screenshot.save(test_screenshot_path)
        logger.info(f"Test screenshot saved: {test_screenshot_path}")
        
        # Create output directory
        output_dir = "test_omniparser_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse screenshot
        ui_elements, labeled_image_path = omniparser.parse_screenshot(
            test_screenshot_path, 
            save_output=True, 
            output_dir=output_dir
        )
        
        logger.info(f"✅ OmniParser Results:")
        logger.info(f"   - Elements detected: {len(ui_elements)}")
        logger.info(f"   - Labeled image: {labeled_image_path}")
        logger.info(f"   - Output directory: {output_dir}")
        
        # Log first few elements
        for i, element in enumerate(ui_elements[:5]):
            content = element.get("content", "")
            interactive = element.get("interactivity", False)
            logger.info(f"   - Element {i+1}: '{content}' ({'Interactive' if interactive else 'Static'})")
        
        # Check if labeled image exists and has content
        if labeled_image_path and os.path.exists(labeled_image_path):
            file_size = os.path.getsize(labeled_image_path)
            logger.info(f"   - Labeled image size: {file_size} bytes")
            if file_size > 10000:  # Reasonable size
                logger.info("   ✅ Labeled image appears to be properly generated")
            else:
                logger.warning("   ⚠ Labeled image might be too small - check annotations")
        else:
            logger.error("   ❌ No labeled image generated")
        
        return len(ui_elements) > 0, labeled_image_path
        
    except Exception as e:
        logger.error(f"❌ OmniParser test failed: {e}")
        return False, None

def test_qwen_logging():
    """Test Qwen response logging."""
    try:
        from qwen import QwenDecisionMaker
        import config
        
        logger.info("Testing Qwen response logging...")
        
        # Initialize Qwen
        qwen = QwenDecisionMaker(config.QWEN_CONFIG)
        
        # Test simple detection
        test_elements = [
            {"content": "OPTIONS", "interactivity": True, "bbox": [0.1, 0.2, 0.3, 0.4]},
            {"content": "EXIT", "interactivity": True, "bbox": [0.5, 0.6, 0.7, 0.8]},
            {"content": "BENCHMARK", "interactivity": True, "bbox": [0.2, 0.3, 0.4, 0.5]}
        ]
        
        # Test benchmark detection
        logger.info("Testing benchmark detection...")
        benchmark_found = qwen.detect_benchmark_results(test_elements)
        logger.info(f"   - Benchmark detection result: {benchmark_found}")
        
        # Test exit detection
        logger.info("Testing exit button detection...")
        exit_button = qwen.find_exit_button(test_elements)
        logger.info(f"   - Exit button result: {exit_button}")
        
        # Check for debug files
        debug_dirs = ["debug_qwen_responses", "debug_flow_analysis"]
        for debug_dir in debug_dirs:
            if os.path.exists(debug_dir):
                files = os.listdir(debug_dir)
                logger.info(f"   - Debug directory '{debug_dir}': {len(files)} files")
                for file in files[-3:]:  # Show last 3 files
                    logger.info(f"     - {file}")
            else:
                logger.info(f"   - Debug directory '{debug_dir}': Not created yet")
        
        logger.info("✅ Qwen logging test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Qwen test failed: {e}")
        return False

def test_configuration():
    """Test configuration settings."""
    try:
        import config
        
        logger.info("Testing configuration...")
        
        # Check OmniParser config
        omni_config = config.OMNIPARSER_CONFIG
        logger.info(f"   - Box threshold: {omni_config.get('box_threshold', 'NOT SET')}")
        logger.info(f"   - IoU threshold: {omni_config.get('iou_threshold', 'NOT SET')}")
        logger.info(f"   - Scale image: {omni_config.get('scale_img', 'NOT SET')}")
        
        # Check debug config
        debug_config = config.DEBUG_CONFIG
        logger.info(f"   - Debug enabled: {debug_config.get('enabled', 'NOT SET')}")
        logger.info(f"   - Save Qwen responses: {debug_config.get('save_qwen_responses', 'NOT SET')}")
        logger.info(f"   - Save OmniParser outputs: {debug_config.get('save_omniparser_outputs', 'NOT SET')}")
        
        # Check model files
        model_files = [
            config.OMNIPARSER_CONFIG.get("icon_detect_model_path", ""),
            config.OMNIPARSER_CONFIG.get("caption_model_path", "")
        ]
        
        for model_file in model_files:
            if model_file and os.path.exists(model_file):
                logger.info(f"   ✅ Model file found: {model_file}")
            else:
                logger.warning(f"   ⚠ Model file missing: {model_file}")
        
        logger.info("✅ Configuration check completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    logger.info("="*60)
    logger.info("AUTOMATED GAME BENCHMARKER - DEBUG VERIFICATION")
    logger.info("="*60)
    
    # Test 1: Configuration
    logger.info("\n1. Testing Configuration...")
    config_ok = test_configuration()
    
    # Test 2: OmniParser Annotations
    logger.info("\n2. Testing OmniParser Annotations...")
    omni_ok, labeled_path = test_omniparser_annotations()
    
    # Test 3: Qwen Logging
    logger.info("\n3. Testing Qwen Logging...")
    qwen_ok = test_qwen_logging()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    logger.info(f"OmniParser Annotations: {'✅ PASS' if omni_ok else '❌ FAIL'}")
    logger.info(f"Qwen Logging: {'✅ PASS' if qwen_ok else '❌ FAIL'}")
    
    if all([config_ok, omni_ok, qwen_ok]):
        logger.info("\n🎉 All tests passed! Your fixes should be working.")
        if labeled_path:
            logger.info(f"📸 Check the labeled image: {labeled_path}")
        logger.info("📁 Check debug directories for Qwen logs")
    else:
        logger.info("\n⚠ Some tests failed. Check the logs above for details.")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()