"""
Updated configuration settings with enhanced button detection fixes.
"""
import os
from datetime import datetime
from pathlib import Path

# Base directory for benchmark runs
BASE_DIR = Path("benchmark_runs")

# ENHANCED OmniParser V2 Configuration - FIXED for better button detection
OMNIPARSER_CONFIG = {
    "icon_detect_model_path": "weights/icon_detect/model.pt",
    "caption_model_name": "florence2",
    "caption_model_path": "weights/icon_caption_florence",
    "device": "cuda",
    "box_threshold": 0.02,  # LOWERED from 0.05 for better detection
    "iou_threshold": 0.4,   # LOWERED from 0.5 for better detection
    "use_local_semantics": True,
    "batch_size": 128,
    "scale_img": False,
    "easyocr_args": {
        "paragraph": False,
        "text_threshold": 0.6  # LOWERED from 0.7 for better text detection
    },
    "use_paddleocr": True
}

# Qwen 2.5 7B Instruct Configuration - OPTIMIZED
QWEN_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "device": "cuda",
    "torch_dtype": "auto",
    "device_map": "auto",
    "temperature": 0.1,
    "top_p": 0.8,
    "max_new_tokens": 256,
    "repetition_penalty": 1.05,
    "do_sample": True
}

# Benchmark execution settings - OPTIMIZED
BENCHMARK_CONFIG = {
    "initial_wait_time": 3,
    "screenshot_interval": 2.0,
    "max_navigation_attempts": 10,
    "benchmark_timeout": 120,
    "benchmark_duration": 70,
    "confidence_threshold": 0.70,
    "navigation_delay": 1.0,
    "result_check_attempts": 3,
    "result_check_interval": 3,
    "analysis_cooldown": 1.5
}

# Flow configuration file
FLOW_CONFIG_FILE = "game_flows.yaml"

# Enhanced decision prompts with button detection awareness
DECISION_PROMPTS = {
    "detect_results": """
Check if benchmark results are shown in these UI elements:
{ui_elements}

Look for: FPS numbers, AVERAGE, MIN, MAX, RESULTS, COMPLETED, FINISHED

Respond: YES or NO
""",

    "find_exit": """
Find exit/quit button from these UI elements:
{ui_elements}

Look for: EXIT, QUIT, EXIT GAME, QUIT GAME, EXIT TO DESKTOP

Response: CLICK: [element_name] or NONE
""",

    "confirmation_dialog": """
This appears to be a confirmation dialog. Look for confirmation buttons:
{ui_elements}

PRIORITY: Find CONFIRM, YES, OK buttons to proceed.
AVOID: CANCEL, NO, BACK buttons unless specifically needed.

Response: CLICK: [element_name] or WAIT
"""
}

# ENHANCED Debug settings - ALWAYS FORCED ON
DEBUG_CONFIG = {
    "enabled": True,
    "verbose_logging": True,
    "save_screenshots": True,
    "save_omniparser_outputs": True,
    "save_qwen_responses": True,
    "draw_bounding_boxes": False,
    "cleanup_old_screenshots": True,
    "enhanced_button_detection": True,  # NEW: Enable button enhancement
    "force_qwen_calls": True,  # NEW: Force Qwen call for every analysis
    "save_enhancement_logs": True  # NEW: Save button enhancement details
}

# Input control settings - OPTIMIZED
INPUT_CONFIG = {
    "smooth_mouse_movement": False,
    "mouse_movement_steps": 5,
    "click_delay": 0.1,
    "key_press_delay": 0.05,
    "post_action_delay": 1.0
}

# Enhanced button detection settings
BUTTON_DETECTION_CONFIG = {
    "button_keywords": [
        "confirm", "cancel", "yes", "no", "ok", "apply", "save", "close",
        "start", "stop", "pause", "resume", "exit", "quit", "back", "next",
        "continue", "finish", "done", "submit", "send", "go", "run", "begin"
    ],
    "confirmation_keywords": ["confirm", "yes", "ok"],
    "cancellation_keywords": ["cancel", "no", "back"],
    "max_button_text_length": 20,  # Text longer than this is unlikely to be a button
    "force_interactive_for_keywords": True,
    "position_based_detection": True  # Use position to identify buttons
}

# Keyboard shortcuts
KEYBOARD_SHORTCUTS = {
    "exit_game": "alt+f4",
    "escape_menu": "escape",
    "screenshot": "f12",
    "enter": "enter"
}

def enhance_ui_elements_interactivity(ui_elements):
    """
    Post-process UI elements to fix misclassified buttons.
    This is the CRITICAL fix for the button detection issue.
    """
    button_keywords = BUTTON_DETECTION_CONFIG["button_keywords"]
    max_length = BUTTON_DETECTION_CONFIG["max_button_text_length"]
    
    enhanced_elements = []
    enhancement_count = 0
    
    for element in ui_elements:
        enhanced_element = element.copy()
        content = element.get("content", "").lower().strip()
        
        # Force buttons to be interactive based on content
        if not enhanced_element.get("interactivity", False):  # Only enhance non-interactive elements
            for keyword in button_keywords:
                if keyword in content and len(content) <= max_length:
                    enhanced_element["interactivity"] = True
                    enhanced_element["element_type"] = "button"
                    enhanced_element["enhancement_reason"] = f"Forced interactive due to keyword: {keyword}"
                    enhancement_count += 1
                    break
            
            # Special handling for exact matches of confirmation/cancellation
            if content in BUTTON_DETECTION_CONFIG["confirmation_keywords"] + BUTTON_DETECTION_CONFIG["cancellation_keywords"]:
                enhanced_element["interactivity"] = True
                enhanced_element["element_type"] = "button"
                enhanced_element["enhancement_reason"] = "Exact match for dialog button"
                enhancement_count += 1
        
        enhanced_elements.append(enhanced_element)
    
    # Log enhancements if debug enabled
    if DEBUG_CONFIG.get("save_enhancement_logs", True) and enhancement_count > 0:
        import logging
        logger = logging.getLogger("ButtonEnhancement")
        logger.info(f"Enhanced {enhancement_count} elements to fix button detection")
        
        for orig, enh in zip(ui_elements, enhanced_elements):
            if orig.get("interactivity") != enh.get("interactivity"):
                logger.info(f"Enhanced: '{enh.get('content')}' -> Interactive: {enh.get('interactivity')} ({enh.get('enhancement_reason', 'N/A')})")
    
    return enhanced_elements

def create_run_directory(game_name: str = "unknown_game"):
    """Create a game-specific timestamped run directory structure and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / f"{game_name}_{timestamp}"
    
    # Create subdirectories
    directories = {
        "root": run_dir,
        "raw_screenshots": run_dir / "Raw_Screenshots",
        "analyzed_screenshots": run_dir / "Analyzed_Screenshots",
        "logs": run_dir / "Logs",
        "runtime_logs": run_dir / "Logs" / "runtime_logs",
        "omniparser_output": run_dir / "Logs" / "omniparser_output",
        "qwen_prompt": run_dir / "Logs" / "qwen_prompt",
        "qwen_responses": run_dir / "Logs" / "qwen_responses",
        "error_logs": run_dir / "Logs" / "error_logs",
        "benchmark_results": run_dir / "Benchmark_Results",
        "enhancement_logs": run_dir / "Logs" / "enhancement_logs"  # NEW: For button enhancement logs
    }
    
    # Create all directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def get_game_path(game_name):
    """Get the executable path for a specific game from YAML config."""
    try:
        import yaml
        with open(FLOW_CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        
        games = config.get('games', {})
        if game_name in games:
            return games[game_name].get('executable_path')
        return None
    except Exception:
        return None

def update_config(config_dict, updates):
    """Update configuration with new values."""
    config_dict.update(updates)
    return config_dict

def log_configuration_status():
    """Log the current configuration status for debugging."""
    import logging
    logger = logging.getLogger("ConfigStatus")
    
    logger.info("=== ENHANCED CONFIGURATION STATUS ===")
    logger.info(f"OmniParser box_threshold: {OMNIPARSER_CONFIG['box_threshold']} (LOWERED)")
    logger.info(f"OmniParser iou_threshold: {OMNIPARSER_CONFIG['iou_threshold']} (LOWERED)")
    logger.info(f"OmniParser text_threshold: {OMNIPARSER_CONFIG['easyocr_args']['text_threshold']} (LOWERED)")
    logger.info(f"Button detection enabled: {DEBUG_CONFIG['enhanced_button_detection']}")
    logger.info(f"Force Qwen calls: {DEBUG_CONFIG['force_qwen_calls']}")
    logger.info(f"Button keywords: {len(BUTTON_DETECTION_CONFIG['button_keywords'])} configured")
    logger.info("=== CONFIGURATION LOADED ===")

# Initialize configuration logging
if __name__ == "__main__":
    log_configuration_status()