"""
Configuration file for Game Benchmarker
Contains all configurable parameters and settings
"""

import os
import torch

class Config:
    # === MODEL CONFIGURATION ===
    MODEL_PATH = "weights/icon_detect/model.pt"
    CAPTION_MODEL_NAME = "florence2"
    CAPTION_MODEL_PATH = "weights/icon_caption_florence"
    
    # Device configuration - auto-detect or force specific device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"  # Uncomment to force CPU usage
    
    # === OMNIPARSER CONFIGURATION ===
    # Box detection threshold (lower = more sensitive)
    BOX_THRESHOLD = 0.05
    
    # OCR configuration
    OCR_TEXT_THRESHOLD = 0.9
    OCR_USE_PADDLE = True
    OCR_PARAGRAPH_MODE = False
    
    # Image processing settings
    USE_LOCAL_SEMANTICS = True
    IOU_THRESHOLD = 0.7
    SCALE_IMG = False
    BATCH_SIZE = 128
    
    # Bounding box drawing configuration
    BBOX_TEXT_SCALE_FACTOR = 0.8
    BBOX_TEXT_THICKNESS_MIN = 1
    BBOX_TEXT_PADDING_MIN = 1
    BBOX_THICKNESS_MIN = 1
    BOX_OVERLAY_RATIO_DIVISOR = 3200
    
    # === TIMING CONFIGURATION ===
    # Benchmark duration in seconds (1 min 30 seconds default)
    BENCHMARK_TIMEOUT = 90
    
    # How often to take snapshots during operations (seconds)
    SNAPSHOT_INTERVAL = 25
    
    # Maximum time to wait for main menu (seconds)
    MAIN_MENU_WAIT_TIME = 300
    
    # Benchmark running duration (seconds)
    BENCHMARK_DURATION = 70
    
    # Delays for UI interactions
    CLICK_DELAY = 0.5
    MOUSE_DOWN_UP_DELAY = 0.2
    KEY_PRESS_DELAY = 0.1
    MENU_TRANSITION_DELAY = 3
    SHORT_WAIT = 2
    CONFIRMATION_WAIT = 5
    
    # === GAME CONFIGURATION ===
    # Default game path - modify this for your specific game
    DEFAULT_GAME_PATH = r"C:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Far Cry 5\bin\FarCry5.exe"
    
    # Maximum attempts for various operations
    MAX_MENU_NAVIGATION_ATTEMPTS = 10
    MAX_BENCHMARK_OPTIONS_ATTEMPTS = 5
    MAX_BENCHMARK_RESULT_CHECKS = 3
    
    # === DIRECTORY CONFIGURATION ===
    BASE_LOG_DIR = "benchmark_logs"
    SNAPSHOT_DIR = os.path.join(BASE_LOG_DIR, "snapsForOmni")
    PARSED_IMAGES_DIR = os.path.join(BASE_LOG_DIR, "omniParsedImages")
    RUNTIME_LOG_DIR = os.path.join(BASE_LOG_DIR, "runtimeLog")
    BENCHMARK_RESULTS_DIR = os.path.join(BASE_LOG_DIR, "gameBenchmarkSnaps")
    
    # === LOGGING CONFIGURATION ===
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOGGER_NAME = "GameBenchmarker"
    
    # === UI ELEMENT DETECTION PROMPTS ===
    # Main menu indicators - multiple lists that should all be present
    MAIN_MENU_INDICATORS = [
        ["start new game", "new game", "play", "campaign", "continue"],
        ["options", "settings", "configuration"],
        ["exit", "quit", "exit game", "quit game"],
    ]
    
    # Options/Settings menu targets
    OPTIONS_TARGETS = ["options", "settings", "configuration"]
    OPTIONS_FALLBACKS = ["menu", "system"]
    
    # Benchmark-related targets
    BENCHMARK_TARGETS = ["benchmark", "benchmarks", "performance test"]
    BENCHMARK_FALLBACKS = ["test", "performance", "fps test"]
    
    # Graphics menu targets
    GRAPHICS_TARGETS = ["graphics", "display", "video"]
    GRAPHICS_FALLBACKS = ["advanced", "quality"]
    
    # FPS and benchmark result indicators
    FPS_INDICATORS = [
        "fps", "average", "min", "max", "1% low", "0.1% low", 
        "frame rate", "framerate", "frame/s"
    ]
    
    # Navigation elements
    BACK_BUTTONS = ["back", "return", "previous", "cancel", "main menu"]
    CONTINUE_BUTTONS = ["ok", "continue", "back", "return", "done", "main menu"]
    CONTINUE_FALLBACKS = ["next", "finish", "exit"]
    
    # Exit game options
    EXIT_OPTIONS = [
        "exit game", "quit game", "exit to desktop", 
        "quit to desktop", "exit", "quit"
    ]
    
    # Confirmation dialog options
    CONFIRM_OPTIONS = ["yes", "confirm", "ok", "accept", "exit"]
    
    @classmethod
    def get_directories(cls):
        """Return all required directories as a list."""
        return [
            cls.SNAPSHOT_DIR,
            cls.PARSED_IMAGES_DIR,
            cls.RUNTIME_LOG_DIR,
            cls.BENCHMARK_RESULTS_DIR,
        ]
    
    # === DEBUG ANNOTATION CONFIGURATION ===
    # Font settings for annotations
    ANNOTATION_FONT_SIZE = 16
    ANNOTATION_FONT_SIZE_SMALL = 12
    
    # Colors for different element types (RGB tuples)
    ANNOTATION_COLOR_INTERACTIVE = (0, 255, 0)  # Bright green for interactive
    ANNOTATION_COLOR_NON_INTERACTIVE = (255, 165, 0)  # Orange for non-interactive
    ANNOTATION_COLOR_INTERACTIVE_BG = (0, 100, 0, 180)  # Semi-transparent green
    ANNOTATION_COLOR_NON_INTERACTIVE_BG = (100, 60, 0, 180)  # Semi-transparent orange
    ANNOTATION_COLOR_TEXT = (255, 255, 255)  # White text
    
    # Bounding box appearance
    ANNOTATION_BOX_WIDTH = 3
    ANNOTATION_MAX_TEXT_LENGTH = 30
    ANNOTATION_TEXT_PADDING = 4
    
    # Legend settings
    ANNOTATION_LEGEND_WIDTH = 200
    ANNOTATION_LEGEND_MARGIN = 10
    ANNOTATION_LEGEND_LINE_HEIGHT = 20
    ANNOTATION_LEGEND_BG = (0, 0, 0, 200)  # Semi-transparent black
    ANNOTATION_LEGEND_BORDER = (255, 255, 255)  # White border
    
    # Comparison image settings
    ANNOTATION_COMPARISON_GAP = 20  # Gap between original and annotated in side-by-side
    
    # Enable/disable debug annotations
    ENABLE_DEBUG_ANNOTATIONS = True
    SAVE_ELEMENTS_LIST = True  # Also save text file with element details
    CREATE_COMPARISON_IMAGES = False  # Set to True for side-by-side comparisons
    
    @classmethod
    def get_bbox_draw_config(cls, box_overlay_ratio):
        """Generate bounding box drawing configuration based on image size."""
        return {
            "text_scale": cls.BBOX_TEXT_SCALE_FACTOR * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), cls.BBOX_TEXT_THICKNESS_MIN),
            "text_padding": max(int(3 * box_overlay_ratio), cls.BBOX_TEXT_PADDING_MIN),
            "thickness": max(int(3 * box_overlay_ratio), cls.BBOX_THICKNESS_MIN),
        }