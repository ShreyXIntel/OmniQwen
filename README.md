# Automated Game Benchmarking System

An intelligent automated system for running game benchmarks using **OmniParser V2** for UI element detection and **Qwen 2.5 7B Instruct** for decision-making. This system automatically navigates through game menus, runs built-in benchmarks, captures results, and gracefully exits games.

## 🌟 Features

- **Fully Automated**: Complete end-to-end benchmarking without human intervention
- **AI-Powered Navigation**: Uses OmniParser V2 + Qwen 2.5 for intelligent UI understanding
- **Modular Architecture**: Clean, maintainable codebase with separate components
- **Win32API Integration**: Precise mouse and keyboard control using Windows APIs
- **Comprehensive Logging**: Detailed logging and debug information for troubleshooting
- **Result Capture**: Automatic detection and saving of benchmark results
- **Graceful Game Handling**: Proper game launching and exit procedures

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Screenshot    │───▶│   OmniParser V2  │───▶│  UI Elements    │
│   Manager       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             ▼
│ Input Controller│◄───│  UI Analyzer     │◄────┌─────────────────┐
│   (Win32API)    │    │                  │     │ Qwen 2.5 7B     │
└─────────────────┘    └──────────────────┘     │ Instruct        │
         │                       ▲              └─────────────────┘
         ▼                       │
┌─────────────────┐              │
│ Game Interface  │──────────────┘
│                 │
└─────────────────┘
```

## 📦 Installation

### Prerequisites

- Windows 10/11 (required for win32api)
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM (for running AI models)

### Dependencies

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install pillow
pip install pywin32
pip install numpy

# OmniParser V2 dependencies
pip install ultralytics
pip install easyocr
pip install paddlepaddle paddleocr  # Optional, for better OCR
```

### Model Setup

1. **Download OmniParser V2 Models**:
   ```bash
   # Create weights directory
   mkdir -p weights/icon_detect
   mkdir -p weights/icon_caption_florence
   
   # Download models (follow OmniParser V2 installation guide)
   # Place icon detection model at: weights/icon_detect/model.pt
   # Place caption model at: weights/icon_caption_florence/
   ```

2. **Verify Installation**:
   ```bash
   python launcher.py --verify
   ```

## 🚀 Quick Start

### Basic Usage

```bash
# Run verification tests first
python launcher.py --verify

# Run benchmark on a game (assuming game is already running)
python launcher.py --no-launch

# Launch and benchmark a specific game
python launcher.py --game far_cry_6

# Use custom game path
python launcher.py --game-path "C:\Path\To\Your\Game.exe"
```

### Configuration

Edit `config.py` to customize:

- **Game Paths**: Add your game executable paths
- **AI Model Settings**: Adjust temperature, confidence thresholds
- **Benchmark Timing**: Modify timeouts and intervals
- **Debug Options**: Enable/disable logging and output saving

```python
# Example game path configuration
GAME_PATHS = {
    "your_game": r"C:\Path\To\Your\Game.exe",
    "cyberpunk_2077": r"C:\Games\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe"
}

# Example AI configuration
QWEN_CONFIG = {
    "temperature": 0.2,  # Lower = more deterministic
    "confidence_threshold": 0.80  # Minimum confidence for actions
}
```

## 📋 Usage Examples

### Example 1: Complete Automated Flow
```bash
# Launch Far Cry 6 and run complete benchmark
python launcher.py --game far_cry_6 --timeout 300
```

### Example 2: Pre-launched Game
```bash
# Game is already running at main menu
python launcher.py --no-launch
```

### Example 3: Debug Mode
```bash
# Run with verbose logging and debug outputs
python launcher.py --game cyberpunk_2077 --debug
```

## 🎮 Supported Games

The system includes configurations for:

- **Far Cry 6**
- **Cyberpunk 2077** 
- **Black Myth: Wukong Benchmark Tool**
- **Counter-Strike 2**
- **Assassin's Creed Series**

### Adding New Games

1. Add game path to `config.py`:
```python
GAME_PATHS = {
    "your_game": r"C:\Path\To\Game.exe"
}
```

2. Test the system:
```bash
python launcher.py --game your_game --debug
```

3. The AI will automatically learn the game's UI patterns!

## 🔧 Configuration Reference

### Key Configuration Files

- **`config.py`**: Main configuration (model settings, game paths, prompts)
- **`launcher.py`**: Entry point with command-line options

### Important Settings

```python
# Benchmark timing
BENCHMARK_CONFIG = {
    "benchmark_duration": 70,      # Expected benchmark time
    "screenshot_interval": 2.0,    # Time between screenshots
    "max_navigation_attempts": 15, # Max menu navigation tries
    "confidence_threshold": 0.80   # Min AI confidence for actions
}

# Debug options
DEBUG_CONFIG = {
    "verbose_logging": True,           # Detailed logs
    "save_omniparser_outputs": True,  # Save UI detection images
    "save_qwen_responses": True       # Save AI decision logs
}
```

## 📊 Output Structure

After running, the system creates:

```
benchmark_runs/
└── run_YYYYMMDD_HHMMSS/
    ├── Raw_Screenshots/           # All captured screenshots
    ├── OmniParser_Outputs/        # UI detection visualizations
    ├── Qwen_Responses/           # AI decision logs
    ├── Logs/                     # System logs and summaries
    └── Benchmark_Results/        # Final benchmark screenshots
```

## 🔍 Troubleshooting

### Common Issues

1. **"CUDA not available"**
   - Install CUDA-compatible PyTorch
   - System will work on CPU but much slower

2. **"OmniParser models not found"**
   - Download OmniParser V2 models to `weights/` directory
   - Run `python launcher.py --verify` to check

3. **"Game not responding to clicks"**
   - Ensure game is in windowed/borderless mode
   - Check if game requires admin privileges
   - Verify screen scaling is 100%

4. **"Navigation loop detected"**
   - AI automatically handles this by pressing Escape
   - May indicate game UI has changed
   - Check debug logs for UI detection issues

### Debug Mode

Run with `--debug` flag for detailed information:

```bash
python launcher.py --debug --no-launch
```

This enables:
- Verbose logging
- UI detection image saving
- AI decision logging
- Step-by-step execution details

### Verification Tests

```bash
python launcher.py --verify
```

Checks:
- ✅ Dependencies installed
- ✅ GPU/CUDA available  
- ✅ Model files present
- ✅ Configuration valid
- ✅ File permissions

## 🧠 How It Works

### The Flow

1. **Screenshot Capture**: Takes screenshot of current game state
2. **UI Detection**: OmniParser V2 identifies all UI elements and their locations
3. **Decision Making**: Qwen 2.5 analyzes UI elements and decides next action
4. **Action Execution**: win32api performs mouse clicks or keyboard presses
5. **Progress Monitoring**: Continuously monitors for benchmark completion
6. **Result Capture**: Detects and saves benchmark results
7. **Graceful Exit**: Navigates back to menu and exits game

### AI Decision Process

The system uses sophisticated prompts to guide Qwen 2.5:

```
GOAL: Navigate to and start the in-game benchmark

UI ELEMENTS DETECTED:
1. "Graphics Settings" - button (Interactive) [340, 510, 600, 580]
2. "Benchmark" - button (Interactive) [200, 300, 400, 350]
3. "Back" - button (Interactive) [50, 50, 150, 100]

DECISION: CLICK on "Benchmark" (confidence: 0.95)
REASONING: Found direct benchmark button, highest priority target
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Game Support**: Add configurations for new games
- **UI Detection**: Improve element detection accuracy
- **Decision Logic**: Enhance AI decision-making prompts
- **Error Handling**: Add more robust error recovery
- **Performance**: Optimize model loading and inference

## ⚖️ License

This project is released under the Apache 2.0 License. See `LICENSE` file for details.

## 🙏 Acknowledgments

- **OmniParser V2**: For excellent UI element detection
- **Qwen Team**: For the powerful Qwen 2.5 language model
- **Microsoft**: For win32api Windows integration
- **Game Developers**: For including built-in benchmark tools

---

**Note**: This tool is for personal use and benchmarking purposes. Ensure you comply with game terms of service and local laws when using automation tools.