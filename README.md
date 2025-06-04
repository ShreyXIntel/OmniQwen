# Automated Game Benchmarker

A modular computer vision-based system for automatically running in-game benchmarks. The system uses OmniParserV2 to detect and interact with game UI elements, enabling fully automated benchmark execution.

## 🏗️ Architecture

The system is organized into modular components for maintainability and flexibility:

```
├── weights                  # Icon detection and Icon caption goes here from HF
├── config.py                # Configuration and settings
├── detector.py             # Computer vision and image detection
├── input_controller.py     # Mouse and keyboard input handling
├── analyzer.py             # UI element analysis and detection logic
├── debug_annotator.py      # Debug visualization and annotation system
├── game_controller.py      # Game-specific operations and navigation
├── benchmark_manager.py    # Main orchestration and benchmark execution
├── main.py                # Entry point and CLI interface
└── util/
    └── utils.py           # OmniParser utility functions (existing)
```

## 🚀 Features

- **Automated Game Launch**: Starts games and waits for main menu detection
- **Intelligent UI Navigation**: Uses computer vision to navigate through game menus
- **Benchmark Execution**: Automatically runs in-game benchmarks and captures results
- **Result Collection**: Takes screenshots of benchmark results for analysis
- **Debug Annotations**: Comprehensive visual debugging with labeled bounding boxes
- **Element Analysis**: Detailed text reports of detected UI elements
- **Graceful Exit**: Properly exits games through UI or force-close as fallback
- **Comprehensive Logging**: Detailed logs of all operations and detections
- **Modular Design**: Easy to extend for different games and use cases

## 📋 Requirements

- Python 3.8+
- PyTorch (for OmniParser)
- PIL (Pillow)
- win32api/win32con (Windows only)
- ultralytics (YOLO)
- OmniParserV2 dependencies

## ⚙️ Configuration

All settings are centralized in `config.py`. Key configuration options:

### Game Settings
```python
# Path to your game executable
DEFAULT_GAME_PATH = r"C:\Path\To\Your\Game.exe"

# Timing settings
BENCHMARK_DURATION = 70  # seconds
MAIN_MENU_WAIT_TIME = 300  # seconds
SNAPSHOT_INTERVAL = 25  # seconds
```

### Model Settings
```python
# OmniParser model configuration
MODEL_PATH = "weights/icon_detect/model.pt"
CAPTION_MODEL_PATH = "weights/icon_caption_florence"
BOX_THRESHOLD = 0.05  # Detection sensitivity
DEVICE = "cuda"  # or "cpu"
```

### UI Element Detection
```python
# Customize UI element detection prompts
MAIN_MENU_INDICATORS = [
    ["start new game", "new game", "play", "campaign"],
    ["options", "settings", "configuration"],
    ["exit", "quit", "exit game"]
]

BENCHMARK_TARGETS = ["benchmark", "benchmarks", "performance test"]
```

### Debug Annotation Settings
```python
# Enable/disable debug features
ENABLE_DEBUG_ANNOTATIONS = True
SAVE_ELEMENTS_LIST = True
CREATE_COMPARISON_IMAGES = False

# Visual annotation styling
ANNOTATION_COLOR_INTERACTIVE = (0, 255, 0)  # Green for clickable
ANNOTATION_COLOR_NON_INTERACTIVE = (255, 165, 0)  # Orange for static
ANNOTATION_FONT_SIZE = 16
ANNOTATION_BOX_WIDTH = 3
```

## 🎮 Usage

### Quick Start
```bash
# Run with default settings
python main.py

# Use custom game path
python main.py --game-path "C:\Games\YourGame\game.exe"

# Test detection without launching game
python main.py --test-detection

# Test debug annotations (screenshot + visual analysis)
python main.py --test-annotations

# Show system status
python main.py --status
```

### Programmatic Usage
```python
from benchmark_manager import BenchmarkManager

# Initialize and run full benchmark
benchmarker = BenchmarkManager("C:\Path\To\Game.exe")
success = benchmarker.run_full_benchmark_cycle()
```

### Using Individual Components
```python
from detector import VisionDetector
from analyzer import UIAnalyzer
from input_controller import InputController

# Take screenshot and analyze
detector = VisionDetector()
analyzer = UIAnalyzer()

snapshot = detector.take_snapshot()
content, labeled = detector.parse_image(snapshot)

# Check for UI elements
if analyzer.check_for_main_menu(content):
    print("Main menu detected!")

# Find and interact with elements
analyzer.find_and_click_element(content, ["play", "start"])
```

### Debug Annotations
```python
from debug_annotator import DebugAnnotator
from detector import VisionDetector

# Take screenshot and create debug annotations
detector = VisionDetector()
annotator = DebugAnnotator()

snapshot = detector.take_snapshot()
content, _ = detector.parse_image(snapshot)

# Create detailed visual annotations
annotated_path = annotator.annotate_image(snapshot, content)

# Generate detailed text report
annotator.annotate_elements_list(content)

# Create side-by-side comparison
annotator.create_comparison_image(snapshot, annotated_path)
```

## 📁 Output Structure

The system creates organized output directories:

```
benchmark_logs/
├── snapsForOmni/          # Raw screenshots
├── omniParsedImages/      # Processed images with UI element labels
│   ├── debug_annotated_*.png     # Annotated images with visual debugging
│   ├── elements_list_*.txt       # Detailed text reports of detected elements
│   └── comparison_*.png          # Side-by-side original vs annotated (optional)
├── gameBenchmarkSnaps/    # Benchmark result screenshots
└── runtimeLog/            # Detailed execution logs
```

## 🔧 Component Details

### VisionDetector (`detector.py`)
- OmniParser model initialization and management
- Screenshot capture and saving
- Image parsing and UI element detection
- Labeled image generation

### InputController (`input_controller.py`)
- Mouse click operations at specific coordinates
- Keyboard input simulation
- Screen coordinate calculations
- Input timing and delays

### UIAnalyzer (`analyzer.py`)
- UI element content analysis
- Element finding and filtering
- Game-specific UI pattern recognition
- Interactive element detection

### GameController (`game_controller.py`)
- Game process launching
- Menu navigation logic
- Main menu detection
- Graceful game exit handling

### DebugAnnotator (`debug_annotator.py`)
- Comprehensive visual annotation of detected UI elements
- Color-coded bounding boxes (green=interactive, orange=static)
- Element numbering and content labels
- Detection confidence scores and metadata
- Legend and summary information overlay
- Text reports with detailed element information
- Optional side-by-side comparison images

### BenchmarkManager (`benchmark_manager.py`)
- Overall orchestration
- Logging setup and management
- Directory creation
- Full benchmark cycle execution

## 🎯 Customization for Different Games

To adapt the system for a new game:

1. **Update `config.py`**:
   ```python
   DEFAULT_GAME_PATH = r"C:\Path\To\NewGame.exe"
   
   # Customize UI element detection terms
   MAIN_MENU_INDICATORS = [
       ["start", "play", "begin"],
       ["settings", "options"],
       ["exit", "quit"]
   ]
   
   BENCHMARK_TARGETS = ["performance", "fps test", "benchmark"]
   ```

2. **Adjust timing settings** if needed:
   ```python
   BENCHMARK_DURATION = 90  # Longer benchmark
   MAIN_MENU_WAIT_TIME = 180  # Faster loading game
   ```

3. **Extend game-specific logic** in `game_controller.py` if needed

## 🛠️ Troubleshooting

### Common Issues

**Detection not working:**
- Check if OmniParser models are properly loaded
- Verify GPU/CUDA availability if using GPU
- Adjust `BOX_THRESHOLD` in config (lower = more sensitive)

**Game not launching:**
- Verify game path in config
- Check if game requires admin privileges
- Ensure game executable is accessible

**Menu navigation failing:**
- Review detection logs to see what elements are found
- Adjust UI element detection terms in config
- Check if game UI language matches detection terms

**Benchmark not starting:**
- Verify benchmark option detection terms
- Check if game requires specific graphics settings
- Review navigation path to benchmark option

### Debug Mode
Enable detailed logging by setting `LOG_LEVEL = "DEBUG"` in config.py

## 📊 Performance Tuning

- **GPU Usage**: Set `DEVICE = "cuda"` for faster processing
- **Detection Sensitivity**: Adjust `BOX_THRESHOLD` (0.01-0.1 range)
- **Batch Size**: Modify `BATCH_SIZE` based on GPU memory
- **Timing**: Adjust delay constants for different game response times

## 🤝 Contributing

The modular architecture makes it easy to contribute:

- Add new games by updating config prompts
- Extend UI detection capabilities in `analyzer.py`
- Improve input handling in `input_controller.py`
- Add new benchmark types in `benchmark_manager.py`

## 📄 License

This project builds upon OmniParserV2 and follows appropriate licensing for computer vision and automation tools.

---

## 💡 Advanced Usage Examples

### Custom Benchmark Workflow
```python
from benchmark_manager import BenchmarkManager

# Initialize with custom settings
benchmarker = BenchmarkManager("C:\Game.exe")

# Run individual steps
game_ctrl = benchmarker.game_controller

# Launch and wait for menu
if game_ctrl.launch_game():
    menu_found, content = game_ctrl.wait_for_main_menu()
    
    if menu_found:
        # Custom navigation logic here
        if game_ctrl.navigate_to_benchmarks(content):
            # Run benchmark with custom duration
            benchmarker.run_benchmark()
            
        game_ctrl.exit_game()
```

### Multiple Game Testing
```python
games = [
    r"C:\Games\Game1\game1.exe",
    r"C:\Games\Game2\game2.exe",
    r"C:\Games\Game3\game3.exe"
]

results = []
for game_path in games:
    benchmarker = BenchmarkManager(game_path)
    success = benchmarker.run_full_benchmark_cycle()
    results.append((game_path, success))
```

This modular design provides maximum flexibility while maintaining the robust automation capabilities of the original system.
