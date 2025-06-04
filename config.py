"""
Configuration settings for the Game Benchmarking System with OmniParser V2 + Qwen 2.5.
"""
import os
from datetime import datetime
from pathlib import Path

# Base directory for benchmark runs
BASE_DIR = Path("benchmark_runs")

# OmniParser V2 Configuration
OMNIPARSER_CONFIG = {
    "icon_detect_model_path": "weights/icon_detect/model.pt",
    "caption_model_name": "florence2",
    "caption_model_path": "weights/icon_caption_florence",
    "device": "cuda",
    "box_threshold": 0.02,  # Lowered from 0.05 to catch more elements
    "iou_threshold": 0.5,   # Lowered from 0.7 for better detection
    "use_local_semantics": True,
    "batch_size": 128,
    "scale_img": True,      # Changed to True for better processing
    "easyocr_args": {
        "paragraph": False,
        "text_threshold": 0.7  # Lowered from 0.9
    },
    "use_paddleocr": True
}
# Qwen 2.5 7B Instruct Configuration
QWEN_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "device": "cuda",
    "torch_dtype": "auto",
    "device_map": "auto",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.05,
    "do_sample": True
}

# Benchmark execution settings
BENCHMARK_CONFIG = {
    "initial_wait_time": 5,         # Seconds to wait after game launch
    "screenshot_interval": 2.0,     # Seconds between screenshots
    "max_navigation_attempts": 15,  # Maximum attempts to navigate through menus
    "benchmark_timeout": 120,       # Maximum seconds to wait for benchmark to complete
    "benchmark_duration": 70,       # Expected benchmark duration in seconds
    "confidence_threshold": 0.80,   # Minimum confidence for UI actions
    "navigation_delay": 1.2,        # Seconds to wait after navigation action
    "result_check_attempts": 3,     # Number of attempts to find benchmark results
    "result_check_interval": 5      # Seconds between result checks
}

# Game paths (add your game paths here)
GAME_PATHS = {
    "far_cry_6": r"C:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Far Cry 6\bin\FarCry6.exe",
    "black_myth_wukong": r"C:\Program Files (x86)\Steam\steamapps\common\Black Myth Wukong Benchmark Tool\b1_benchmark.exe",
    "cyberpunk_2077": r"C:\Program Files (x86)\Steam\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
    "counter_strike_2": r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\cs2.exe",
    "assassins_creed": r"C:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Assassin's Creed Valhalla\ACValhalla.exe"
}

DEBUG_CONFIG = {
    "enabled": True,
    "verbose_logging": True,
    "save_screenshots": True,
    "save_omniparser_outputs": True,
    "save_qwen_responses": True,  # Ensure this is True
    "draw_bounding_boxes": True,
    "cleanup_old_screenshots": False
}

"""
Configuration settings for the Game Benchmarking System with OmniParser V2 + Qwen 2.5.
"""
import os
from datetime import datetime
from pathlib import Path

# Base directory for benchmark runs
BASE_DIR = Path("benchmark_runs")

# OmniParser V2 Configuration
OMNIPARSER_CONFIG = {
    "icon_detect_model_path": "weights/icon_detect/model.pt",
    "caption_model_name": "florence2",
    "caption_model_path": "weights/icon_caption_florence",
    "device": "cuda",
    "box_threshold": 0.05,
    "iou_threshold": 0.7,
    "use_local_semantics": True,
    "batch_size": 128,
    "scale_img": False,
    "easyocr_args": {
        "paragraph": False,
        "text_threshold": 0.9
    },
    "use_paddleocr": True
}

# Qwen 2.5 7B Instruct Configuration
QWEN_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "device": "cuda",
    "torch_dtype": "auto",
    "device_map": "auto",
    "temperature": 0.1,  # Lower temperature for more consistent decisions
    "top_p": 0.8,
    "max_new_tokens": 512,  # Shorter responses for faster processing
    "repetition_penalty": 1.05,
    "do_sample": True
}

# Benchmark execution settings
BENCHMARK_CONFIG = {
    "initial_wait_time": 5,         # Seconds to wait after game launch
    "screenshot_interval": 2.0,     # Seconds between screenshots
    "max_navigation_attempts": 15,  # Maximum attempts to navigate through menus
    "benchmark_timeout": 120,       # Maximum seconds to wait for benchmark to complete
    "benchmark_duration": 70,       # Expected benchmark duration in seconds
    "confidence_threshold": 0.70,   # Minimum confidence for UI actions (lowered for flow system)
    "navigation_delay": 1.2,        # Seconds to wait after navigation action
    "result_check_attempts": 5,     # Number of attempts to find benchmark results
    "result_check_interval": 5      # Seconds between result checks
}

# Game paths (add your game paths here)
GAME_PATHS = {
    "far_cry_5": r"C:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Far Cry 5\bin\FarCry5.exe",
    "far_cry_6": r"C:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Far Cry 6\bin\FarCry6.exe",
    "black_myth_wukong": r"C:\Program Files (x86)\Steam\steamapps\common\Black Myth Wukong Benchmark Tool\b1_benchmark.exe",
    "cyberpunk_2077": r"C:\Program Files (x86)\Steam\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
    "counter_strike_2": r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\cs2.exe",
    "assassins_creed": r"C:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Assassin's Creed Valhalla\ACValhalla.exe"
}

# Flow configuration file
FLOW_CONFIG_FILE = "game_flows.yaml"

# Simplified decision prompts for basic functions (kept for compatibility)
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
"""
}

# Debug settings
DEBUG_CONFIG = {
    "enabled": True,
    "verbose_logging": True,
    "save_screenshots": True,
    "save_omniparser_outputs": True,
    "save_qwen_responses": True,
    "draw_bounding_boxes": True,
    "cleanup_old_screenshots": False
}

# Input control settings
INPUT_CONFIG = {
    "smooth_mouse_movement": True,
    "mouse_movement_steps": 8,
    "click_delay": 0.2,
    "key_press_delay": 0.1,
    "post_action_delay": 1.5
}

# Keyboard shortcuts
KEYBOARD_SHORTCUTS = {
    "exit_game": "alt+f4",
    "escape_menu": "escape",
    "screenshot": "f12",
    "enter": "enter"
}

def create_run_directory():
    """Create a timestamped run directory structure and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / f"run_{timestamp}"
    
    # Create subdirectories
    directories = {
        "root": run_dir,
        "raw_screenshots": run_dir / "Raw_Screenshots",
        "omniparser_outputs": run_dir / "OmniParser_Outputs", 
        "qwen_responses": run_dir / "Qwen_Responses",
        "logs": run_dir / "Logs",
        "benchmark_results": run_dir / "Benchmark_Results"
    }
    
    # Create all directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def get_game_path(game_name):
    """Get the executable path for a specific game."""
    return GAME_PATHS.get(game_name.lower(), None)

def update_config(config_dict, updates):
    """Update configuration with new values."""
    config_dict.update(updates)
    return config_dict

# Debug settings
DEBUG_CONFIG = {
    "enabled": True,
    "verbose_logging": True,
    "save_screenshots": True,
    "save_omniparser_outputs": True,
    "save_qwen_responses": True,
    "draw_bounding_boxes": True
}

# Input control settings
INPUT_CONFIG = {
    "smooth_mouse_movement": True,
    "mouse_movement_steps": 10,
    "click_delay": 0.2,
    "key_press_delay": 0.1,
    "post_action_delay": 1.0
}

# Keyboard shortcuts
KEYBOARD_SHORTCUTS = {
    "exit_game": "alt+f4",
    "escape_menu": "escape",
    "screenshot": "f12",
    "enter": "enter"
}

def create_run_directory():
    """Create a timestamped run directory structure and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / f"run_{timestamp}"
    
    # Create subdirectories
    directories = {
        "root": run_dir,
        "raw_screenshots": run_dir / "Raw_Screenshots",
        "omniparser_outputs": run_dir / "OmniParser_Outputs", 
        "qwen_responses": run_dir / "Qwen_Responses",
        "logs": run_dir / "Logs",
        "benchmark_results": run_dir / "Benchmark_Results"
    }
    
    # Create all directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def get_game_path(game_name):
    """Get the executable path for a specific game."""
    return GAME_PATHS.get(game_name.lower(), None)

def update_config(config_dict, updates):
    """Update configuration with new values."""
    config_dict.update(updates)
    return config_dict