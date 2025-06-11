"""
Adaptive State-Based Flow Executor - Fixed version to resolve state detection and clicking issues.
"""
import os
import logging
import time
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("AdaptiveFlowExecutor")

class BenchmarkState(Enum):
    """Possible states during benchmark flow."""
    UNKNOWN = "unknown"
    SPLASH_SCREEN = "splash_screen"
    MAIN_MENU = "main_menu"
    OPTIONS_MENU = "options_menu"
    GRAPHICS_SETTINGS = "graphics_settings"
    BENCHMARK_MENU = "benchmark_menu"
    BENCHMARK_CONFIRM = "benchmark_confirm"
    BENCHMARK_RUNNING = "benchmark_running"
    BENCHMARK_RESULTS = "benchmark_results"
    EXIT_CONFIRM = "exit_confirm"

@dataclass
class StateAction:
    """Action to take in a given state."""
    action: str  # click, sleep, wait, press_key
    target: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    sleep_duration: Optional[int] = None

class AdaptiveFlowExecutor:
    """State-based flow executor that adapts to current game state."""
    
    def __init__(self, flow_file_path: str = "game_flows.yaml"):
        self.flow_file_path = flow_file_path
        self.flows = {}
        self.current_game = None
        self.current_state = BenchmarkState.UNKNOWN
        self.goal_state = BenchmarkState.BENCHMARK_RESULTS
        self.max_attempts_per_state = 3
        self.state_attempt_count = 0
        self.total_attempts = 0
        self.max_total_attempts = 20
        
        # Enhanced state detection keywords with better confirmation dialog detection
        self.state_keywords = {
            BenchmarkState.SPLASH_SCREEN: ["press any button", "click anywhere", "loading", "black myth", "wukong"],
            BenchmarkState.MAIN_MENU: ["benchmark", "settings", "options", "exit", "new game", "continue"],
            BenchmarkState.OPTIONS_MENU: ["graphics", "video", "audio", "controls", "gameplay"],
            BenchmarkState.GRAPHICS_SETTINGS: ["benchmark", "ray tracing", "quality", "resolution"],
            BenchmarkState.BENCHMARK_MENU: ["start", "run", "begin", "confirm"],
            # FIXED: Better keywords for confirmation dialog detection
            BenchmarkState.BENCHMARK_CONFIRM: [
                "do you want", "are you sure", "confirm", "cancel", "yes", "no", 
                "start benchmark", "run benchmark", "begin test", "proceed"
            ],
            BenchmarkState.BENCHMARK_RUNNING: ["loading", "testing", "running", "progress"],
            BenchmarkState.BENCHMARK_RESULTS: ["results", "score", "fps", "average", "minimum", "maximum", "finished"],
            BenchmarkState.EXIT_CONFIRM: ["quit", "exit", "yes", "no", "cancel"]
        }
        
        # Track state history to prevent loops
        self.state_history = []
        self.last_successful_state = BenchmarkState.UNKNOWN
        
        self._load_flows()
        logger.info("Adaptive Flow Executor initialized")
    
    def set_debug_directory(self, debug_dir: str):
        """Set debug directory for logging compatibility."""
        self.debug_base_dir = debug_dir
    
    def _load_flows(self):
        """Load flow configuration for reference."""
        try:
            if os.path.exists(self.flow_file_path):
                with open(self.flow_file_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.flows = config.get('games', {})
                logger.info(f"Loaded flow references for {len(self.flows)} games")
            else:
                logger.error(f"Flow file not found: {self.flow_file_path}")
        except Exception as e:
            logger.error(f"Failed to load flows: {e}")
    
    def set_game(self, game_name: str) -> bool:
        """Set current game."""
        if game_name in self.flows:
            self.current_game = game_name
            self.current_state = BenchmarkState.UNKNOWN
            self.state_attempt_count = 0
            self.total_attempts = 0
            self.state_history = []
            self.last_successful_state = BenchmarkState.UNKNOWN
            logger.info(f"Set game to: {game_name}")
            return True
        return False
    
    def detect_current_state(self, ui_elements: List[Dict]) -> BenchmarkState:
        """Detect current state based on UI elements with improved confirmation dialog detection."""
        if not ui_elements:
            return BenchmarkState.UNKNOWN
        
        # Extract all text content from UI elements
        element_texts = []
        interactive_elements = []
        
        for element in ui_elements:
            content = element.get("content", "").lower().strip()
            if content:
                element_texts.append(content)
                if element.get("interactivity", False):
                    interactive_elements.append(content)
        
        all_text = " ".join(element_texts)
        interactive_text = " ".join(interactive_elements)
        
        logger.debug(f"Detecting state from text: {all_text[:200]}...")
        logger.debug(f"Interactive elements: {interactive_text[:100]}...")
        
        # Score each state based on keyword matches with enhanced scoring
        state_scores = {}
        for state, keywords in self.state_keywords.items():
            score = 0
            keyword_matches = []
            
            for keyword in keywords:
                # Check in all text
                if keyword.lower() in all_text:
                    score += 1
                    keyword_matches.append(keyword)
                
                # Boost score for interactive elements
                if keyword.lower() in interactive_text:
                    score += 2
                    logger.debug(f"Interactive match: '{keyword}' for state {state.value}")
            
            # Special handling for confirmation dialog detection
            if state == BenchmarkState.BENCHMARK_CONFIRM:
                # Look for confirmation dialog patterns
                confirmation_patterns = [
                    "do you want",
                    "are you sure", 
                    "start benchmark",
                    "confirm" in interactive_text and "cancel" in interactive_text
                ]
                
                for pattern in confirmation_patterns:
                    if isinstance(pattern, str) and pattern in all_text:
                        score += 3  # High score for confirmation patterns
                        keyword_matches.append(pattern)
                        logger.info(f"Confirmation dialog pattern detected: '{pattern}'")
                    elif isinstance(pattern, bool) and pattern:
                        score += 3
                        keyword_matches.append("confirm+cancel buttons")
                        logger.info("Confirm and Cancel buttons detected together")
            
            state_scores[state] = score
            if score > 0:
                logger.debug(f"State {state.value}: score={score}, matches={keyword_matches}")
        
        # Find the state with highest score
        best_state = max(state_scores, key=state_scores.get)
        best_score = state_scores[best_state]
        
        if best_score > 0:
            # Additional validation for state transitions
            if self._validate_state_transition(self.current_state, best_state, ui_elements):
                logger.info(f"Detected state: {best_state.value} (score: {best_score})")
                self._record_state_history(best_state)
                return best_state
            else:
                logger.warning(f"Invalid state transition from {self.current_state.value} to {best_state.value}")
                return self.current_state
        else:
            logger.warning("Could not detect state - returning current state")
            return self.current_state
    
    def _validate_state_transition(self, from_state: BenchmarkState, to_state: BenchmarkState, ui_elements: List[Dict]) -> bool:
        """Validate if a state transition makes sense."""
        # Allow staying in same state
        if from_state == to_state:
            return True
        
        # Define valid transitions
        valid_transitions = {
            BenchmarkState.UNKNOWN: [BenchmarkState.SPLASH_SCREEN, BenchmarkState.MAIN_MENU, BenchmarkState.BENCHMARK_CONFIRM],
            BenchmarkState.SPLASH_SCREEN: [BenchmarkState.MAIN_MENU],
            BenchmarkState.MAIN_MENU: [BenchmarkState.BENCHMARK_CONFIRM, BenchmarkState.OPTIONS_MENU, BenchmarkState.GRAPHICS_SETTINGS],
            BenchmarkState.OPTIONS_MENU: [BenchmarkState.GRAPHICS_SETTINGS, BenchmarkState.MAIN_MENU],
            BenchmarkState.GRAPHICS_SETTINGS: [BenchmarkState.BENCHMARK_MENU, BenchmarkState.OPTIONS_MENU],
            BenchmarkState.BENCHMARK_MENU: [BenchmarkState.BENCHMARK_CONFIRM],
            BenchmarkState.BENCHMARK_CONFIRM: [BenchmarkState.BENCHMARK_RUNNING, BenchmarkState.MAIN_MENU],
            BenchmarkState.BENCHMARK_RUNNING: [BenchmarkState.BENCHMARK_RESULTS],
            BenchmarkState.BENCHMARK_RESULTS: [BenchmarkState.MAIN_MENU, BenchmarkState.EXIT_CONFIRM],
            BenchmarkState.EXIT_CONFIRM: [BenchmarkState.SPLASH_SCREEN]
        }
        
        allowed_transitions = valid_transitions.get(from_state, [])
        return to_state in allowed_transitions
    
    def _record_state_history(self, state: BenchmarkState):
        """Record state in history for loop detection."""
        self.state_history.append({
            'state': state,
            'timestamp': time.time(),
            'attempt': self.total_attempts
        })
        
        # Keep only recent history
        if len(self.state_history) > 10:
            self.state_history = self.state_history[-10:]
    
    def decide_next_action(self, current_state: BenchmarkState, ui_elements: List[Dict], 
                          ui_analyzer, directories: Dict) -> StateAction:
        """Decide what action to take based on current state with improved element detection."""
        
        logger.info(f"Deciding action for state: {current_state.value}")
        
        # State-specific decision logic
        if current_state == BenchmarkState.SPLASH_SCREEN:
            return self._handle_splash_screen(ui_elements)
        
        elif current_state == BenchmarkState.MAIN_MENU:
            return self._handle_main_menu(ui_elements, ui_analyzer, directories)
        
        elif current_state == BenchmarkState.OPTIONS_MENU:
            return self._handle_options_menu(ui_elements, ui_analyzer, directories)
        
        elif current_state == BenchmarkState.GRAPHICS_SETTINGS:
            return self._handle_graphics_settings(ui_elements, ui_analyzer, directories)
        
        elif current_state == BenchmarkState.BENCHMARK_MENU:
            return self._handle_benchmark_menu(ui_elements, ui_analyzer, directories)
        
        elif current_state == BenchmarkState.BENCHMARK_CONFIRM:
            return self._handle_benchmark_confirm(ui_elements, ui_analyzer, directories)
        
        elif current_state == BenchmarkState.BENCHMARK_RUNNING:
            return self._handle_benchmark_running()
        
        elif current_state == BenchmarkState.BENCHMARK_RESULTS:
            return self._handle_benchmark_results(ui_elements)
        
        else:  # UNKNOWN or other states
            return self._handle_unknown_state(ui_elements, ui_analyzer, directories)
    
    def _handle_benchmark_confirm(self, ui_elements: List[Dict], ui_analyzer, directories: Dict) -> StateAction:
        """Handle benchmark confirmation state with improved button detection."""
        logger.info("Handling benchmark confirmation dialog")
        
        # Enhanced confirmation button detection
        confirmation_buttons = []
        cancel_buttons = []
        
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
                
            content = element.get("content", "").lower().strip()
            logger.debug(f"Checking element: '{content}'")
            
            # Look for confirmation buttons with exact and partial matching
            confirm_keywords = ["confirm", "yes", "ok", "start", "proceed", "continue"]
            cancel_keywords = ["cancel", "no", "back"]
            
            # Check for confirmation buttons
            for keyword in confirm_keywords:
                if keyword in content:
                    confirmation_buttons.append(element)
                    logger.info(f"Found confirmation button: '{content}' (matched: {keyword})")
                    break
            
            # Check for cancel buttons
            for keyword in cancel_keywords:
                if keyword in content:
                    cancel_buttons.append(element)
                    logger.info(f"Found cancel button: '{content}' (matched: {keyword})")
                    break
        
        # Priority 1: Use dedicated confirmation button
        if confirmation_buttons:
            best_confirm = confirmation_buttons[0]  # Take first/best match
            logger.info(f"Clicking confirmation button: '{best_confirm.get('content')}'")
            return StateAction("click", best_confirm.get("content"), 0.9, "Found dedicated confirmation button")
        
        # Priority 2: Look for buttons that might be confirm buttons by position/context
        all_buttons = [e for e in ui_elements if e.get("interactivity")]
        
        # If we have exactly 2 buttons, assume right/bottom one is confirm
        if len(all_buttons) == 2:
            # Sort by x-coordinate (left to right) or y-coordinate (top to bottom)
            try:
                buttons_sorted = sorted(all_buttons, key=lambda x: (x.get("bbox", [0, 0, 0, 0])[0], x.get("bbox", [0, 0, 0, 0])[1]))
                
                # Usually confirm is on the right or bottom
                confirm_button = buttons_sorted[1] if len(buttons_sorted) > 1 else buttons_sorted[0]
                logger.info(f"Using positional logic for confirmation: '{confirm_button.get('content')}'")
                return StateAction("click", confirm_button.get("content"), 0.7, "Using positional logic for confirmation")
            except Exception as e:
                logger.warning(f"Error in positional sorting: {e}")
        
        # Priority 3: Look for any button that's not clearly a cancel button
        for element in all_buttons:
            content = element.get("content", "").lower()
            if not any(cancel_word in content for cancel_word in ["cancel", "no", "back", "exit"]):
                logger.info(f"Using non-cancel button as confirmation: '{element.get('content')}'")
                return StateAction("click", element.get("content"), 0.6, "Using non-cancel button")
        
        # Priority 4: If we find text mentioning benchmark, click anywhere in that area
        for element in ui_elements:
            content = element.get("content", "").lower()
            if "benchmark" in content and "start" in content:
                logger.info(f"Clicking on benchmark start text: '{element.get('content')}'")
                return StateAction("click", element.get("content"), 0.5, "Clicking on benchmark start text")
        
        # Fallback: Wait and try again
        logger.warning("Could not find clear confirmation button, waiting...")
        return StateAction("wait", None, 0.3, "No clear confirmation button found")
    
    def _handle_splash_screen(self, ui_elements: List[Dict]) -> StateAction:
        """Handle splash screen state."""
        # Look for "Press Any Button" or similar
        for element in ui_elements:
            content = element.get("content", "").lower()
            if element.get("interactivity") and any(keyword in content for keyword in ["press", "click", "any", "button"]):
                return StateAction("click", element.get("content"), 0.9, "Found interactive prompt on splash screen")
        
        # If no specific button, click anywhere
        return StateAction("click_anywhere", None, 0.7, "Splash screen - clicking anywhere to proceed")
    
    def _handle_main_menu(self, ui_elements: List[Dict], ui_analyzer, directories: Dict) -> StateAction:
        """Handle main menu state."""
        # Priority 1: Look for direct benchmark button
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
            content = element.get("content", "").lower()
            if "benchmark" in content:
                return StateAction("click", element.get("content"), 1.0, "Found direct benchmark button in main menu")
        
        # Priority 2: Look for settings/options to access benchmark
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
            content = element.get("content", "").lower()
            if any(keyword in content for keyword in ["settings", "options", "graphics", "video"]):
                return StateAction("click", element.get("content"), 0.8, "Going to settings to find benchmark")
        
        return StateAction("wait", None, 0.3, "No clear path found in main menu")
    
    def _handle_options_menu(self, ui_elements: List[Dict], ui_analyzer, directories: Dict) -> StateAction:
        """Handle options menu state."""
        # Look for graphics/video settings
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
            content = element.get("content", "").lower()
            if any(keyword in content for keyword in ["graphics", "video", "display"]):
                return StateAction("click", element.get("content"), 0.9, "Going to graphics settings for benchmark")
        
        # Look for direct benchmark option
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
            content = element.get("content", "").lower()
            if "benchmark" in content:
                return StateAction("click", element.get("content"), 1.0, "Found benchmark in options menu")
        
        return StateAction("press_key", "escape", 0.5, "No relevant options found, going back")
    
    def _handle_graphics_settings(self, ui_elements: List[Dict], ui_analyzer, directories: Dict) -> StateAction:
        """Handle graphics settings state."""
        # Look for benchmark option
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
            content = element.get("content", "").lower()
            if "benchmark" in content:
                return StateAction("click", element.get("content"), 1.0, "Found benchmark in graphics settings")
        
        return StateAction("press_key", "escape", 0.4, "No benchmark found in graphics settings")
    
    def _handle_benchmark_menu(self, ui_elements: List[Dict], ui_analyzer, directories: Dict) -> StateAction:
        """Handle benchmark menu state."""
        # Look for start/run buttons
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
            content = element.get("content", "").lower()
            if any(keyword in content for keyword in ["start", "run", "begin", "go"]):
                return StateAction("click", element.get("content"), 0.9, "Starting benchmark")
        
        return StateAction("wait", None, 0.3, "Looking for start button")
    
    def _handle_benchmark_running(self) -> StateAction:
        """Handle benchmark running state."""
        # Get game-specific sleep time
        sleep_time = 90  # Default
        if self.current_game and self.current_game in self.flows:
            sleep_time = self.flows[self.current_game].get('benchmark_sleep_time', 90)
        
        return StateAction("sleep", None, 1.0, f"Benchmark running - sleeping for {sleep_time}s", sleep_time)
    
    def _handle_benchmark_results(self, ui_elements: List[Dict]) -> StateAction:
        """Handle benchmark results state."""
        # We've reached our goal!
        return StateAction("wait", None, 1.0, "Benchmark results detected - goal achieved!")
    
    def _handle_unknown_state(self, ui_elements: List[Dict], ui_analyzer, directories: Dict) -> StateAction:
        """Handle unknown state - try to find benchmark-related elements."""
        # Use fuzzy search for benchmark-related terms
        benchmark_keywords = ["benchmark", "test", "performance"]
        
        # Look for any benchmark-related interactive elements
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
            content = element.get("content", "").lower()
            
            # Fuzzy matching for benchmark terms
            for keyword in benchmark_keywords:
                if self._fuzzy_match(keyword, content):
                    return StateAction("click", element.get("content"), 0.7, f"Fuzzy match: {content} -> {keyword}")
        
        # Look for navigation options
        nav_keywords = ["options", "settings", "menu", "graphics", "video"]
        for element in ui_elements:
            if not element.get("interactivity"):
                continue
            content = element.get("content", "").lower()
            for keyword in nav_keywords:
                if keyword in content:
                    return StateAction("click", element.get("content"), 0.5, f"Navigating via {content}")
        
        return StateAction("wait", None, 0.2, "Unknown state - waiting for more information")
    
    def _fuzzy_match(self, target: str, content: str, threshold: float = 0.6) -> bool:
        """Simple fuzzy matching."""
        if target in content or content in target:
            return True
        
        # Character intersection method
        target_chars = set(target.lower())
        content_chars = set(content.lower())
        if target_chars and content_chars:
            intersection = len(target_chars.intersection(content_chars))
            union = len(target_chars.union(content_chars))
            similarity = intersection / union if union > 0 else 0
            return similarity >= threshold
        
        return False
    
    def execute_adaptive_flow(self, ui_analyzer, input_controller, screenshot_manager,
                            screen_width: int, screen_height: int, directories: Dict) -> bool:
        """Execute adaptive flow that can start from any state with fixed debug output saving."""
        try:
            logger.info("=== STARTING ADAPTIVE BENCHMARK FLOW ===")
            
            while self.total_attempts < self.max_total_attempts:
                self.total_attempts += 1
                
                # Take screenshot with unique name for each attempt
                screenshot_path = screenshot_manager.take_screenshot(
                    custom_name=f"adaptive_attempt_{self.total_attempts:02d}"
                )
                if not screenshot_path:
                    logger.error("Failed to take screenshot")
                    continue
                
                # FIXED: Parse UI elements with debug output saving for EVERY analysis
                ui_elements, labeled_image_path = ui_analyzer.omniparser.parse_screenshot(
                    screenshot_path, 
                    save_output=True,  # Always save debug output
                    output_dir=directories.get("analyzed_screenshots")
                )
                
                # FIXED: Force save analysis debug info for every attempt
                if hasattr(ui_analyzer, '_save_analysis_debug'):
                    ui_analyzer._save_analysis_debug(ui_elements, screenshot_path, directories, f"attempt_{self.total_attempts:02d}")
                
                # Detect current state
                detected_state = self.detect_current_state(ui_elements)
                
                # Check if state changed (reset attempt counter if so)
                if detected_state != self.current_state:
                    logger.info(f"State transition: {self.current_state.value} -> {detected_state.value}")
                    self.current_state = detected_state
                    self.state_attempt_count = 0
                    self.last_successful_state = detected_state
                
                self.state_attempt_count += 1
                
                # Check if we've reached our goal
                if self.current_state == BenchmarkState.BENCHMARK_RESULTS:
                    logger.info("SUCCESS: Reached benchmark results!")
                    return True
                
                # FIXED: Better loop detection - check for state loops
                if self._detect_state_loop():
                    logger.warning("State loop detected, attempting to break out...")
                    # Try pressing escape to reset
                    input_controller.press_key("escape")
                    time.sleep(3)
                    self.state_attempt_count = 0
                    continue
                
                # Check if we've tried this state too many times
                if self.state_attempt_count > self.max_attempts_per_state:
                    logger.warning(f"Max attempts reached for state {self.current_state.value}")
                    # Try pressing escape to reset, but don't reset counter immediately
                    input_controller.press_key("escape")
                    time.sleep(2)
                    continue
                
                # Decide next action based on current state
                action = self.decide_next_action(self.current_state, ui_elements, ui_analyzer, directories)
                
                logger.info(f"State: {self.current_state.value}, Action: {action.action}, Target: {action.target}, Confidence: {action.confidence:.2f}")
                logger.info(f"Reasoning: {action.reasoning}")
                
                # Execute the action
                success = self._execute_action(action, ui_elements, input_controller, screen_width, screen_height)
                
                if success:
                    # Wait for action to take effect
                    if action.action == "sleep":
                        logger.info(f"Benchmark sleep completed - {action.sleep_duration}s")
                        # After sleep, we should be at results
                        self.current_state = BenchmarkState.BENCHMARK_RESULTS
                    else:
                        time.sleep(2)  # Normal action cooldown
                else:
                    logger.warning(f"Action failed: {action.action}")
                    time.sleep(1)
            
            logger.error(f"Max total attempts ({self.max_total_attempts}) reached")
            return False
            
        except Exception as e:
            logger.error(f"Error in adaptive flow: {e}")
            return False
    
    def _detect_state_loop(self) -> bool:
        """Detect if we're stuck in a state loop."""
        if len(self.state_history) < 6:
            return False
        
        # Check if we're cycling between the same states
        recent_states = [entry['state'] for entry in self.state_history[-6:]]
        
        # Simple pattern detection - if we see the same state 3+ times in recent history
        state_counts = {}
        for state in recent_states:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        max_count = max(state_counts.values()) if state_counts else 0
        if max_count >= 3:
            logger.warning(f"State loop detected: {state_counts}")
            return True
        
        return False
    
    def _execute_action(self, action: StateAction, ui_elements: List[Dict], input_controller,
                       screen_width: int, screen_height: int) -> bool:
        """Execute the decided action."""
        try:
            if action.action == "click" and action.target:
                # Find the element to click
                target_element = None
                for element in ui_elements:
                    if element.get("content", "").strip() == action.target.strip():
                        target_element = element
                        break
                
                if target_element:
                    return input_controller.click_element(target_element, screen_width, screen_height)
                else:
                    logger.error(f"Target element not found: {action.target}")
                    return False
            
            elif action.action == "click_anywhere":
                return input_controller.click_at_coordinates(screen_width // 2, screen_height // 2)
            
            elif action.action == "press_key":
                return input_controller.press_key(action.target or "escape")
            
            elif action.action == "sleep":
                sleep_duration = action.sleep_duration or 90
                logger.info(f"Sleeping for {sleep_duration} seconds (benchmark running)")
                
                # Sleep with progress updates
                for i in range(0, sleep_duration, 10):
                    actual_sleep = min(10, sleep_duration - i)
                    time.sleep(actual_sleep)
                    progress = ((i + actual_sleep) / sleep_duration) * 100
                    logger.info(f"Benchmark progress: {progress:.1f}% ({i + actual_sleep}/{sleep_duration}s)")
                
                return True
            
            elif action.action == "wait":
                time.sleep(3)
                return True
            
            else:
                logger.error(f"Unknown action: {action.action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action.action}: {e}")
            return False
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive flow execution."""
        return {
            "game": self.current_game,
            "current_state": self.current_state.value,
            "total_attempts": self.total_attempts,
            "state_attempt_count": self.state_attempt_count,
            "goal_reached": self.current_state == BenchmarkState.BENCHMARK_RESULTS,
            "max_attempts": self.max_total_attempts,
            "state_history": [entry['state'].value for entry in self.state_history[-5:]],
            "last_successful_state": self.last_successful_state.value
        }