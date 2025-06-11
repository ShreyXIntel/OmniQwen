"""
Optimized Flow Executor with fuzzy matching, sleep handling, and flexible step management.
"""
import os
import logging
import time
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache
import difflib

logger = logging.getLogger("FlowExecutor")

@dataclass
class FlowStep:
    """Flow step representation with sleep support."""
    step: int
    screen_context: str
    description: str
    expected_buttons: List[str]
    target_button: str
    action: str
    max_retries: int
    success_indicators: List[str]
    sleep_duration: Optional[int] = None

@dataclass
class FlowResult:
    """Flow result representation."""
    success: bool
    attempts: int
    final_action: Optional[str]
    error_message: Optional[str]
    detected_elements: List[str]

class FlowExecutor:
    """Enhanced flow executor with fuzzy matching and sleep support."""
    
    def __init__(self, flow_file_path: str = "game_flows.yaml"):
        """Initialize with fuzzy matching capabilities."""
        self.flow_file_path = flow_file_path
        self.flows = {}
        self.current_game = None
        self.current_step = 0
        self.step_history = []
        self.debug_base_dir = None
        self.keyword_variations = {}
        self.fuzzy_match_threshold = 0.6  # Default threshold
        
        # Performance optimizations
        self._keyword_cache = {}
        self._element_cache = {}
        self._fuzzy_cache = {}
        
        # Load flows once
        self._load_flows()
        logger.info("Flow Executor initialized with fuzzy matching support")
    
    def _load_flows(self):
        """Load flows from YAML file."""
        try:
            if os.path.exists(self.flow_file_path):
                with open(self.flow_file_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.flows = config.get('games', {})
                    self.exit_flow = config.get('exit_flow', {})
                    self.result_detection = config.get('result_detection', {})
                    self.settings = config.get('settings', {})
                    
                    # Load fuzzy match threshold from settings
                    self.fuzzy_match_threshold = self.settings.get('fuzzy_match_threshold', 0.6)
                
                logger.info(f"Loaded flows for {len(self.flows)} games")
                logger.info(f"Fuzzy match threshold: {self.fuzzy_match_threshold}")
            else:
                logger.error(f"Flow file not found: {self.flow_file_path}")
                self.flows = {}
        except Exception as e:
            logger.error(f"Failed to load flows: {e}")
            self.flows = {}
    
    def set_debug_directory(self, debug_dir: str):
        """Set debug directory for logging."""
        self.debug_base_dir = debug_dir
    
    def set_game(self, game_name: str) -> bool:
        """Set current game and reset state."""
        if game_name in self.flows:
            self.current_game = game_name
            self.current_step = 0
            self.step_history = []
            self._keyword_cache.clear()
            self._element_cache.clear()
            self._fuzzy_cache.clear()
            
            # Load keyword variations for this game
            game_config = self.flows[game_name]
            self.keyword_variations = game_config.get('keyword_variations', {})
            
            logger.info(f"Set current game to: {game_name}")
            return True
        else:
            logger.error(f"No flow found for game: {game_name}")
            return False
    
    def get_current_step_info(self) -> Optional[FlowStep]:
        """Get current step information with sleep support."""
        if not self.current_game or self.current_game not in self.flows:
            return None
        
        game_flow = self.flows[self.current_game]['flow']
        if self.current_step >= len(game_flow):
            return None
        
        step_data = game_flow[self.current_step]
        return FlowStep(
            step=step_data['step'],
            screen_context=step_data['screen_context'],
            description=step_data['description'],
            expected_buttons=step_data['expected_buttons'],
            target_button=step_data['target_button'],
            action=step_data['action'],
            max_retries=step_data['max_retries'],
            success_indicators=step_data['success_indicators'],
            sleep_duration=step_data.get('sleep_duration', None)
        )
    
    def execute_current_step(self, ui_analyzer, input_controller, screenshot_manager, 
                           screen_width: int, screen_height: int, directories: Dict = None) -> FlowResult:
        """Execute current step with fuzzy matching and sleep support."""
        step_info = self.get_current_step_info()
        if not step_info:
            return FlowResult(False, 0, None, "No current step available", [])
        
        logger.info(f"Executing Step {step_info.step}: {step_info.description}")
        
        # Handle sleep action directly
        if step_info.action.lower() == "sleep":
            return self._execute_sleep_step(step_info)
        
        attempts = 0
        last_error = None
        detected_elements = []
        
        # Reduce retries for better performance
        max_retries = min(step_info.max_retries, 2)
        
        for attempt in range(max_retries):
            attempts += 1
            logger.info(f"Step {step_info.step} - Attempt {attempt + 1}/{max_retries}")
            
            try:
                # Take screenshot
                screenshot_path = screenshot_manager.take_screenshot(
                    custom_name=f"step_{step_info.step:02d}_attempt_{attempt+1:02d}"
                )
                if not screenshot_path:
                    last_error = "Failed to take screenshot"
                    continue
                
                # Analyze UI
                analysis_result = ui_analyzer.analyze_screenshot(
                    screenshot_path, directories, step_info.screen_context
                )
                
                ui_elements = analysis_result["ui_elements"]
                detected_elements = [e.get("content", "") for e in ui_elements]
                
                # Find target element with fuzzy matching
                target_element = self._find_target_element_fuzzy(ui_elements, step_info)
                
                if target_element:
                    logger.info(f"Found target element: '{target_element.get('content', 'Unknown')}' (fuzzy match)")
                    
                    # Execute action
                    success = self._execute_action(
                        step_info, target_element, input_controller, 
                        screen_width, screen_height
                    )
                    
                    if success:
                        # Quick verification (reduced wait time)
                        time.sleep(1.0)
                        
                        # Verify success
                        if self._verify_success_quick(ui_analyzer, screenshot_manager, step_info):
                            self._record_step_history(step_info, True, attempts, None)
                            return FlowResult(True, attempts, step_info.action, None, detected_elements)
                    
                    last_error = "Action executed but success not verified"
                else:
                    last_error = f"Target '{step_info.target_button}' not found (even with fuzzy matching)"
                    logger.warning(f"Detected elements: {detected_elements}")
                
            except Exception as e:
                last_error = f"Exception: {str(e)}"
                logger.error(f"Exception in step execution: {e}")
            
            # Reduced wait between retries
            if attempt < max_retries - 1:
                time.sleep(1.0)
        
        # Step failed
        self._record_step_history(step_info, False, attempts, last_error)
        return FlowResult(False, attempts, None, last_error, detected_elements)
    
    def _execute_sleep_step(self, step_info: FlowStep) -> FlowResult:
        """Execute sleep step for benchmark waiting."""
        sleep_duration = step_info.sleep_duration
        if not sleep_duration:
            logger.error("Sleep step missing sleep_duration")
            return FlowResult(False, 1, None, "Sleep duration not specified", [])
        
        logger.info(f"Starting sleep step: waiting {sleep_duration} seconds for benchmark to complete")
        
        # Log progress every 10 seconds
        total_sleep = sleep_duration
        interval = 10
        elapsed = 0
        
        while elapsed < total_sleep:
            remaining_chunk = min(interval, total_sleep - elapsed)
            time.sleep(remaining_chunk)
            elapsed += remaining_chunk
            
            progress_pct = (elapsed / total_sleep) * 100
            logger.info(f"Benchmark sleep progress: {progress_pct:.1f}% ({elapsed}/{total_sleep}s)")
        
        logger.info(f"Sleep step completed - benchmark should be finished after {total_sleep} seconds")
        
        # Record successful sleep step
        self._record_step_history(step_info, True, 1, None)
        return FlowResult(True, 1, "sleep", None, [f"Slept for {sleep_duration} seconds"])
    
    def _find_target_element_fuzzy(self, ui_elements: List[Dict], step_info: FlowStep) -> Optional[Dict]:
        """Find target element with fuzzy matching capabilities."""
        # Check cache first
        cache_key = f"{step_info.target_button}_{step_info.screen_context}"
        if cache_key in self._fuzzy_cache:
            cached_content = self._fuzzy_cache[cache_key]
            for element in ui_elements:
                if element.get("content", "").upper() == cached_content:
                    return element
        
        target_upper = step_info.target_button.upper()
        
        # Direct exact match first
        for element in ui_elements:
            content = element.get("content", "").upper()
            if content == target_upper:
                self._fuzzy_cache[cache_key] = content
                return element
        
        # Fuzzy matching for corrupted/partial text
        best_match = None
        best_score = 0
        
        for element in ui_elements:
            content = element.get("content", "").upper()
            if not content:
                continue
            
            # Calculate similarity score
            similarity = self._calculate_fuzzy_similarity(target_upper, content)
            
            if similarity > best_score and similarity >= self.fuzzy_match_threshold:
                best_score = similarity
                best_match = element
        
        if best_match:
            logger.info(f"Fuzzy match found: '{best_match.get('content', '')}' (score: {best_score:.2f}) for target '{step_info.target_button}'")
            self._fuzzy_cache[cache_key] = best_match.get("content", "").upper()
            return best_match
        
        # Keyword variations fallback
        if self.keyword_variations:
            for key, variations in self.keyword_variations.items():
                if target_upper == key.upper():
                    for variation in variations:
                        for element in ui_elements:
                            content = element.get("content", "").upper()
                            
                            # Exact variation match
                            if variation.upper() == content:
                                self._fuzzy_cache[cache_key] = content
                                return element
                            
                            # Fuzzy variation match
                            similarity = self._calculate_fuzzy_similarity(variation.upper(), content)
                            if similarity >= self.fuzzy_match_threshold:
                                logger.info(f"Fuzzy variation match: '{content}' (score: {similarity:.2f}) for variation '{variation}'")
                                self._fuzzy_cache[cache_key] = content
                                return element
        
        # Expected buttons fallback with fuzzy matching
        for expected in step_info.expected_buttons:
            for element in ui_elements:
                content = element.get("content", "").upper()
                
                # Exact expected match
                if expected.upper() == content:
                    self._fuzzy_cache[cache_key] = content
                    return element
                
                # Fuzzy expected match
                similarity = self._calculate_fuzzy_similarity(expected.upper(), content)
                if similarity >= self.fuzzy_match_threshold:
                    logger.info(f"Fuzzy expected match: '{content}' (score: {similarity:.2f}) for expected '{expected}'")
                    self._fuzzy_cache[cache_key] = content
                    return element
        
        return None
    
    def _calculate_fuzzy_similarity(self, target: str, content: str) -> float:
        """Calculate fuzzy similarity between target and content strings."""
        if not target or not content:
            return 0.0
        
        # Use multiple similarity methods and take the best score
        scores = []
        
        # Method 1: difflib.SequenceMatcher (ratio)
        scores.append(difflib.SequenceMatcher(None, target, content).ratio())
        
        # Method 2: Substring matching
        if target in content or content in target:
            scores.append(min(len(target), len(content)) / max(len(target), len(content)))
        
        # Method 3: Character intersection ratio
        target_chars = set(target)
        content_chars = set(content)
        if target_chars or content_chars:
            intersection = len(target_chars.intersection(content_chars))
            union = len(target_chars.union(content_chars))
            scores.append(intersection / union if union > 0 else 0)
        
        # Method 4: Common subsequence
        common_subseq = self._longest_common_subsequence(target, content)
        if common_subseq > 0:
            scores.append(common_subseq / max(len(target), len(content)))
        
        # Return the best score
        return max(scores) if scores else 0.0
    
    def _longest_common_subsequence(self, s1: str, s2: str) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(s1), len(s2)
        if m == 0 or n == 0:
            return 0
        
        # Use dynamic programming for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _execute_action(self, step_info: FlowStep, target_element: Dict,
                       input_controller, screen_width: int, screen_height: int) -> bool:
        """Execute step action including sleep support."""
        try:
            action = step_info.action.lower()
            
            if action == "click":
                return input_controller.click_element(target_element, screen_width, screen_height)
            elif action == "click_anywhere":
                return input_controller.click_at_coordinates(screen_width // 2, screen_height // 2)
            elif action == "press_escape":
                return input_controller.press_key("escape")
            elif action == "press_enter":
                return input_controller.press_key("enter")
            elif action == "sleep":
                # This should be handled by _execute_sleep_step, but as fallback
                sleep_duration = step_info.sleep_duration or 30
                logger.info(f"Executing sleep action for {sleep_duration} seconds")
                time.sleep(sleep_duration)
                return True
            else:
                logger.error(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return False
    
    def _verify_success_quick(self, ui_analyzer, screenshot_manager, step_info: FlowStep) -> bool:
        """Quick success verification with fuzzy matching for indicators."""
        try:
            # Skip verification for sleep steps
            if step_info.action.lower() == "sleep":
                return True
            
            # Take verification screenshot
            screenshot_path = screenshot_manager.take_screenshot(
                custom_name=f"verify_step_{step_info.step:02d}"
            )
            if not screenshot_path:
                return False
            
            # Quick parse without saving
            ui_elements, _ = ui_analyzer.omniparser.parse_screenshot(
                screenshot_path, save_output=False
            )
            
            # Check success indicators with fuzzy matching
            for element in ui_elements:
                element_text = element.get("content", "").upper()
                for indicator in step_info.success_indicators:
                    indicator_upper = indicator.upper()
                    
                    # Exact match
                    if indicator_upper in element_text:
                        logger.info(f"Success indicator found (exact): {indicator}")
                        return True
                    
                    # Fuzzy match for indicators
                    similarity = self._calculate_fuzzy_similarity(indicator_upper, element_text)
                    if similarity >= self.fuzzy_match_threshold:
                        logger.info(f"Success indicator found (fuzzy): {indicator} -> {element_text} (score: {similarity:.2f})")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verifying success: {e}")
            return False
    
    def _record_step_history(self, step_info: FlowStep, success: bool, attempts: int, error: Optional[str]):
        """Record step in history."""
        history_entry = {
            'step': step_info.step,
            'success': success,
            'attempts': attempts,
            'action': step_info.action,
            'target': step_info.target_button,
            'time': time.time(),
            'sleep_duration': step_info.sleep_duration
        }
        
        if error:
            history_entry['error'] = error
        
        self.step_history.append(history_entry)
        
        # Keep only recent history
        if len(self.step_history) > 15:
            self.step_history = self.step_history[-15:]
    
    def advance_step(self) -> bool:
        """Advance to next step."""
        if not self.current_game or self.current_game not in self.flows:
            return False
        
        game_flow = self.flows[self.current_game]['flow']
        if self.current_step < len(game_flow) - 1:
            self.current_step += 1
            return True
        return False
    
    def is_flow_complete(self) -> bool:
        """Check if flow is complete."""
        if not self.current_game or self.current_game not in self.flows:
            return True
        
        game_flow = self.flows[self.current_game]['flow']
        return self.current_step >= len(game_flow)
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get flow execution summary."""
        if not self.current_game:
            return {"error": "No current game set"}
        
        total_steps = len(self.flows[self.current_game]['flow'])
        completed_steps = len([h for h in self.step_history if h.get('success', False)])
        sleep_steps = len([h for h in self.step_history if h.get('action') == 'sleep' and h.get('success', False)])
        
        return {
            "game": self.current_game,
            "total_steps": total_steps,
            "current_step": self.current_step + 1,
            "completed_steps": completed_steps,
            "sleep_steps_executed": sleep_steps,
            "is_complete": self.is_flow_complete(),
            "success_rate": completed_steps / max(len(self.step_history), 1),
            "fuzzy_match_threshold": self.fuzzy_match_threshold
        }
    
    def get_game_executable_path(self, game_name: str) -> Optional[str]:
        """Get executable path for a game."""
        if game_name in self.flows:
            return self.flows[game_name].get('executable_path')
        return None
    
    def get_game_benchmark_sleep_time(self, game_name: str) -> Optional[int]:
        """Get benchmark sleep time for a game."""
        if game_name in self.flows:
            return self.flows[game_name].get('benchmark_sleep_time')
        return None
    
    def detect_benchmark_results(self, ui_analyzer, screenshot_manager) -> bool:
        """Quick benchmark result detection with fuzzy matching."""
        try:
            screenshot_path = screenshot_manager.take_screenshot()
            if not screenshot_path:
                return False
            
            ui_elements, _ = ui_analyzer.omniparser.parse_screenshot(
                screenshot_path, save_output=False
            )
            
            # Quick check for result indicators with fuzzy matching
            fps_indicators = self.result_detection.get('fps_indicators', [])
            completion_indicators = self.result_detection.get('completion_indicators', [])
            
            indicator_count = 0
            for element in ui_elements:
                element_text = element.get("content", "").upper()
                
                for indicator in fps_indicators + completion_indicators:
                    indicator_upper = indicator.upper()
                    
                    # Exact match
                    if indicator_upper in element_text:
                        indicator_count += 1
                        if indicator_count >= 2:
                            return True
                    
                    # Fuzzy match
                    similarity = self._calculate_fuzzy_similarity(indicator_upper, element_text)
                    if similarity >= self.fuzzy_match_threshold:
                        indicator_count += 1
                        if indicator_count >= 2:
                            logger.info(f"Fuzzy result detection: {indicator} -> {element_text} (score: {similarity:.2f})")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting benchmark results: {e}")
            return False