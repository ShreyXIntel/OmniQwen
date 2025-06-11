# Enhanced Automated Game Benchmarking System

An intelligent, **game-agnostic** automated system for running game benchmarks using **OmniParser V2** for UI element detection and **Qwen 2.5 7B Instruct** for flow-aware decision-making. This enhanced system automatically navigates through any game's menus using YAML-defined configurations, runs built-in benchmarks, captures results, and gracefully exits games.

## 🆕 What's New in the Enhanced Version

### 🎯 Game-Agnostic State Management
- **YAML-Based Configuration**: State detection keywords moved from hardcoded Python to flexible YAML files
- **Works Across Different Games**: Same codebase handles Far Cry, Cyberpunk, Counter-Strike, F1, Horizon, and more
- **Flow Progression Awareness**: Understands expected flow sequences and adapts navigation accordingly
- **Intelligent State Detection**: Uses game-specific keywords to accurately identify current UI state

### 🧠 Enhanced AI Decision Making
- **Flow-Aware Qwen Prompts**: AI understands where you are in the benchmark flow
- **Context-Sensitive Navigation**: Makes better decisions based on expected next states
- **Game-Specific Understanding**: Knows terminology and UI patterns for each game
- **Adaptive Reasoning**: Can handle unexpected states and navigation loops

### 🔍 Comprehensive Debug Logging
- **Bulletproof Output Saving**: All debug files are guaranteed to be created
- **Enhanced Analysis Summaries**: Detailed breakdowns of each decision
- **Flow Progression Tracking**: See exactly where you are in the benchmark flow
- **Error Recovery Logging**: Comprehensive error tracking and recovery attempts

## 🌟 Key Features

### Universal Game Support
- **Game-Agnostic Architecture**: Add new games by editing YAML, no code changes needed
- **Flow Definition System**: Define expected UI progression for any game
- **Priority Keyword System**: Game-specific navigation helpers and confirmation words
- **State Transition Maps**: Understand how states connect in each game's UI

### Intelligent Navigation
- **Adaptive State Machine**: Can start from any game state and find the benchmark
- **Flow Context Understanding**: Knows where it should go next based on current position
- **Loop Detection & Recovery**: Automatically breaks out of navigation loops
- **Fuzzy Text Matching**: Handles OCR errors and UI text variations

### Enhanced Debugging
- **Multi-Format Outputs**: Both human-readable text and machine-readable JSON
- **Comprehensive Screenshots**: Raw, analyzed, and result screenshots with metadata
- **AI Decision Tracking**: Every Qwen prompt and response saved for analysis
- **Flow Execution Logs**: Detailed progression through each game's flow

## 🏗️ Enhanced Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   YAML Config   │───▶│  Adaptive Flow   │───▶│ State Detection │
│  (Game-Agnostic)│    │   Executor       │    │ (Game-Specific) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
┌─────────────────┐    ┌──────────────────┐             ▼
│ Enhanced Qwen   │◄───│  Enhanced UI     │◄────┌─────────────────┐
│ (Flow-Aware)    │    │   Analyzer       │     │  Game Context   │
└─────────────────┘    └──────────────────┘     │  Understanding  │
         │                       ▲              └─────────────────┘
         ▼                       │
┌─────────────────┐              │
│ Smart Navigation│──────────────┘
│ (Context-Aware) │
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

2. **Verify Enhanced Installation**:
   ```bash
   python launcher.py --verify
   ```

## 🚀 Quick Start

### Enhanced Usage Examples

```bash
# Run enhanced verification tests first
python launcher.py --verify

# Run benchmark with game-agnostic detection (assumes game running)
python launcher.py --game far_cry_6

# Works with any configured game using same codebase
python launcher.py --game cyberpunk_2077
python launcher.py --game counter_strike_2
python launcher.py --game f1_22
python launcher.py --game horizon_zero_dawn

# Show enhanced game information
python launcher.py --game-info far_cry_6

# List all games with enhanced features
python launcher.py --list-games
```

### Configuration

The enhanced system uses **YAML configuration** instead of hardcoded values:

```yaml
# game_flows.yaml - Enhanced Configuration
games:
  your_game:
    name: "Your Game Name"
    
    # High-level flow definition
    flow_states:
      sequence: ["MAIN_MENU", "OPTIONS_MENU", "GRAPHICS_SETTINGS", "BENCHMARK_RUNNING", "BENCHMARK_RESULTS"]
      goal_state: "BENCHMARK_RESULTS"
      transitions:
        MAIN_MENU: ["OPTIONS_MENU"]
        OPTIONS_MENU: ["GRAPHICS_SETTINGS", "MAIN_MENU"]
        # ... etc
    
    # Game-specific state detection keywords
    state_keywords:
      MAIN_MENU: ["continue", "new game", "options", "settings", "exit"]
      OPTIONS_MENU: ["graphics", "video", "audio", "controls", "gameplay"]
      GRAPHICS_SETTINGS: ["benchmark", "display", "quality", "resolution"]
      BENCHMARK_RESULTS: ["results", "score", "fps", "average", "finished"]
    
    # Priority keywords for navigation
    priority_keywords:
      benchmark_triggers: ["benchmark", "performance test", "graphics test"]
      navigation_helpers: ["options", "settings", "graphics"]
      confirmation_words: ["yes", "confirm", "ok", "start"]
```

## 🎮 Supported Games (Enhanced)

The enhanced system includes comprehensive configurations for:

- **Far Cry 5** ✨ Enhanced
- **Far Cry 6** ✨ Enhanced  
- **Cyberpunk 2077** ✨ Enhanced
- **Counter-Strike 2** ✨ Enhanced
- **F1 22** ✨ Enhanced
- **Horizon Zero Dawn** ✨ Enhanced
- **Black Myth: Wukong Benchmark Tool** ✨ Enhanced

### Adding New Games (Enhanced Method)

1. **Add game configuration to `game_flows.yaml`**:
```yaml
your_new_game:
  name: "Your New Game"
  flow_states:
    sequence: ["MAIN_MENU", "SETTINGS", "BENCHMARK_RUNNING", "BENCHMARK_RESULTS"]
    goal_state: "BENCHMARK_RESULTS"
  state_keywords:
    MAIN_MENU: ["play", "options", "exit"]
    # ... add keywords for each state
  priority_keywords:
    benchmark_triggers: ["benchmark", "test"]
    # ... add navigation helpers
```

2. **Test the enhanced system**:
```bash
python launcher.py --game your_new_game --debug
```

3. **The AI automatically learns the game's UI patterns!** No code changes needed.

## 🔧 Enhanced Configuration Reference

### Game Configuration Structure

```yaml
game_name:
  # Basic info
  name: "Display Name"
  benchmark_duration: 70
  benchmark_sleep_time: 75
  
  # Enhanced flow definition
  flow_states:
    sequence: ["STATE1", "STATE2", "STATE3"]  # Expected progression
    goal_state: "BENCHMARK_RESULTS"           # Target state
    transitions:                              # Valid state transitions
      STATE1: ["STATE2"]
      STATE2: ["STATE3", "STATE1"]
  
  # Game-specific detection keywords
  state_keywords:
    MAIN_MENU: ["continue", "new game", "options"]
    OPTIONS_MENU: ["graphics", "audio", "controls"]
    # ... for each state in your flow
  
  # Navigation assistance
  priority_keywords:
    benchmark_triggers: ["benchmark", "test", "performance"]
    navigation_helpers: ["options", "settings", "graphics"]
    confirmation_words: ["yes", "confirm", "ok", "start"]
```

### Enhanced Debug Configuration

```python
# config.py - Enhanced debugging
DEBUG_CONFIG = {
    "enabled": True,                    # Always enabled in enhanced version
    "verbose_logging": True,           # Detailed flow progression logs
    "save_screenshots": True,          # Raw + analyzed screenshots
    "save_omniparser_outputs": True,  # UI detection results
    "save_qwen_responses": True       # AI decision logs
}
```

## 📊 Enhanced Output Structure

The enhanced system creates comprehensive debug outputs:

```
benchmark_runs/
└── game_name_YYYYMMDD_HHMMSS/
    ├── Raw_Screenshots/              # All captured screenshots
    ├── Analyzed_Screenshots/         # UI detection visualizations  
    ├── Benchmark_Results/           # Final benchmark screenshots
    └── Logs/
        ├── runtime_logs/            # Main execution logs
        │   ├── benchmark.log        # Primary log file
        │   ├── enhanced_session_summary.json
        │   └── analysis_summary_*.txt
        ├── omniparser_output/       # UI detection details
        │   ├── omniparser_*.txt     # Human-readable
        │   └── omniparser_*.json    # Machine-readable
        ├── qwen_prompt/            # AI prompts sent
        ├── qwen_responses/         # AI decisions made
        └── error_logs/             # Error tracking
```

## 🔍 Enhanced Troubleshooting

### Understanding Flow Progression

Check the enhanced logs to see exactly where the system is in the benchmark flow:

```
Flow Progress: 2/5 (OPTIONS_MENU -> GRAPHICS_SETTINGS)
Expected Next States: GRAPHICS_SETTINGS, MAIN_MENU
Current Confidence: 0.85
State Detection: OPTIONS_MENU (score: 4.0)
```

### Debug Analysis Steps

1. **Check Flow Execution**:
   ```bash
   python launcher.py --game-info your_game
   ```

2. **Verify State Detection**:
   - Look at `omniparser_output/` for UI element detection
   - Check `qwen_responses/` for AI reasoning

3. **Analyze Navigation Decisions**:
   - Review `runtime_logs/analysis_summary_*.txt` for step-by-step decisions
   - Check flow progression in main log file

### Common Issues & Solutions

1. **"State detection confidence low"**
   - Add more specific keywords to `state_keywords` in YAML
   - Check if OCR is detecting UI text correctly in `omniparser_output/`

2. **"Navigation loop detected"**
   - Review flow transitions in YAML configuration
   - Check if expected next states are correctly defined

3. **"Benchmark option not found"**
   - Verify `benchmark_triggers` keywords match your game's terminology
   - Check screenshots to see actual UI text detected

4. **"Flow progression stuck"**
   - Review `flow_states.sequence` to ensure correct order
   - Check if state transitions allow proper navigation

## 🧪 Enhanced Verification

Run comprehensive tests to verify all enhanced features:

```bash
python launcher.py --verify
```

This checks:
- ✅ Dependencies installed
- ✅ Enhanced YAML configuration valid
- ✅ Game-agnostic state detection working
- ✅ Flow progression system functional
- ✅ Debug output system operational

## 🔄 Migration from Legacy Version

If you have the original version:

1. **Update to enhanced YAML**: State keywords moved from Python to YAML
2. **Add flow definitions**: Define expected UI progression for each game
3. **Update launch commands**: Use new `--game` parameter with enhanced features
4. **Review debug outputs**: Enhanced logging provides much more detail

## 🤝 Contributing

The enhanced system makes contributions easier:

### Adding Game Support
- **No code changes needed**: Just edit YAML configuration
- **Flow definition**: Define the expected UI progression
- **Keyword mapping**: Map UI text to internal states
- **Test & validate**: Enhanced verification tests ensure everything works

### Improving AI Decision Making
- **Enhanced prompts**: Flow-aware templates in `qwen.py`
- **Context understanding**: Game-specific reasoning in adaptive executor
- **Error handling**: Robust fallback mechanisms

### Debugging & Analysis
- **Comprehensive logging**: Multiple output formats for analysis
- **Flow visualization**: See exactly how navigation progresses
- **Performance metrics**: Detailed timing and success rate tracking

## ⚖️ License

This enhanced project is released under the Apache 2.0 License. See `LICENSE` file for details.

## 🙏 Acknowledgments

- **OmniParser V2**: For excellent UI element detection capabilities
- **Qwen Team**: For the powerful Qwen 2.5 language model with context understanding
- **Microsoft**: For win32api Windows integration
- **Game Developers**: For including built-in benchmark tools
- **Community**: For feedback that drove these enhancements

## 📈 Enhancement Summary

| Feature | Original Version | Enhanced Version |
|---------|------------------|------------------|
| **Game Support** | Hardcoded for specific games | Game-agnostic YAML configuration |
| **State Detection** | Fixed keywords in Python | Flexible keywords per game in YAML |
| **Navigation** | Rigid step-by-step | Flow-aware adaptive navigation |
| **AI Context** | Basic UI analysis | Flow progression understanding |
| **Debug Output** | Basic logs | Comprehensive multi-format outputs |
| **New Game Addition** | Code changes required | YAML editing only |
| **Error Recovery** | Limited | Intelligent loop detection & recovery |
| **Flow Understanding** | None | High-level progression awareness |

---

**Enhanced Note**: This system now works intelligently across different game UIs using YAML-defined configurations. The same codebase can handle Far Cry, Cyberpunk, Counter-Strike, and any other game you configure - just by editing YAML files, no programming required!