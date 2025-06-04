"""
Flow Executor for YAML-based game navigation flows.
"""
import os
import logging
import time
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger("FlowExecutor")

@dataclass
class FlowStep:
    """Represents a single step in the game flow."""
    step: int
    screen_context: str
    description: str
    expected_buttons: List[str]
    target_button: str
    action: str
    max_retries: int
    success_indicators: List[str]

@dataclass
class FlowResult:
    """Result of executing a flow step."""
    success: bool
    attempts: int
    final_action: Optional[str]
    error_message: Optional[str]
    detected_elements: List[str]

class FlowExecutor:
    """Executes game-specific navigation flows based on YAML configuration."""
    
    def __init__(self, flow_file_path: str = "game_flows.yaml"):
        """Initialize the flow executor.
        
        Args:
            flow_file_path: Path to the YAML flow configuration file
        """
        self.flow_file_path = flow_file_path
        self.flows = {}
        self.current_game = None
        self.current_step = 0
        self.step_history = []
        
        self._load_flows()
        logger.info("Flow Executor initialized")
    
    def _load_flows(self):
        """Load flow configurations from YAML file."""
        try:
            if os.path.exists(self.flow_file_path):
                with open(self.flow_file_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.flows = config.get('games', {})
                    self.exit_flow = config.get('exit_flow', {})
                    self.result_detection = config.get('result_detection', {})
                    self.settings = config.get('settings', {})
                
                logger.info(f"Loaded flows for {len(self.flows)} games")
                for game_name in self.flows.keys():
                    step_count = len(self.flows[game_name].get('flow', []))
                    logger.info(f"  {game_name}: {step_count} steps")
            else:
                logger.error(f"Flow file not found: {self.flow_file_path}")
                self.flows = {}
        except Exception as e:
            logger.error(f"Failed to load flows: {e}")
            self.flows = {}
    
    def set_game(self, game_name: str) -> bool:
        """Set the current game and reset flow state.
        
        Args:
            game_name: Name of the game (must match YAML key)
            
        Returns:
            True if game flow found, False otherwise
        """
        if game_name in self.flows:
            self.current_game = game_name
            self.current_step = 0
            self.step_history = []
            logger.info(f"Set current game to: {game_name}")
            return True
        else:
            logger.error(f"No flow found for game: {game_name}")
            logger.info(f"Available games: {list(self.flows.keys())}")
            return False
    
    def get_current_step_info(self) -> Optional[FlowStep]:
        """Get information about the current step.
        
        Returns:
            FlowStep object or None if no current step
        """
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
            success_indicators=step_data['success_indicators']
        )
    
    def execute_current_step(self, ui_analyzer, input_controller, screenshot_manager, 
                           screen_width: int, screen_height: int, directories: Dict = None) -> FlowResult:
        """Execute the current step with retry logic.
        
        Args:
            ui_analyzer: UI analyzer instance
            input_controller: Input controller instance
            screenshot_manager: Screenshot manager instance
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            FlowResult with execution details
        """
        step_info = self.get_current_step_info()
        if not step_info:
            return FlowResult(False, 0, None, "No current step available", [])
        
        logger.info(f"Executing Step {step_info.step}: {step_info.description}")
        logger.info(f"Target: {step_info.target_button} (Action: {step_info.action})")
        
        attempts = 0
        last_error = None
        detected_elements = []
        
        for attempt in range(step_info.max_retries):
            attempts += 1
            logger.info(f"Step {step_info.step} - Attempt {attempt + 1}/{step_info.max_retries}")
            
            try:
                # Take screenshot
                screenshot_path = screenshot_manager.take_screenshot()
                if not screenshot_path:
                    last_error = "Failed to take screenshot"
                    continue
                
                # Analyze UI with context
                analysis_result = self._analyze_step_context(
                    ui_analyzer, screenshot_path, step_info, directories
                )
                
                if analysis_result["found_elements"]:
                    detected_elements = analysis_result["found_elements"]
                
                # Execute action based on step configuration
                action_success = self._execute_step_action(
                    step_info, analysis_result, input_controller, 
                    screen_width, screen_height
                )
                
                if action_success:
                    # Wait for action to take effect
                    cooldown = self.settings.get('navigation_cooldown', 1.5)
                    time.sleep(cooldown)
                    
                    # Verify success by checking for success indicators
                    if self._verify_step_success(
                        ui_analyzer, screenshot_manager, step_info
                    ):
                        logger.info(f"Step {step_info.step} completed successfully!")
                        self.step_history.append({
                            'step': step_info.step,
                            'success': True,
                            'attempts': attempts,
                            'action': step_info.action
                        })
                        return FlowResult(True, attempts, step_info.action, None, detected_elements)
                    else:
                        last_error = f"Action executed but success indicators not found"
                        logger.warning(f"Action executed but success not verified (attempt {attempt + 1})")
                else:
                    last_error = f"Failed to execute action: {step_info.action}"
                    logger.warning(f"Failed to execute action (attempt {attempt + 1})")
                
            except Exception as e:
                last_error = f"Exception during execution: {str(e)}"
                logger.error(f"Exception in step execution: {e}")
            
            # Wait before retry (except for last attempt)
            if attempt < step_info.max_retries - 1:
                retry_delay = 2.0
                logger.info(f"Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)
        
        # All attempts failed
        logger.error(f"Step {step_info.step} failed after {attempts} attempts")
        self.step_history.append({
            'step': step_info.step,
            'success': False,
            'attempts': attempts,
            'error': last_error
        })
        
        return FlowResult(False, attempts, None, last_error, detected_elements)
    
    def _save_step_analysis_debug(self, step_info: FlowStep, ui_elements: List[Dict], 
                            qwen_analysis: Dict, screenshot_path: str):
        """Save comprehensive step analysis for debugging."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            debug_dir = "debug_flow_analysis"
            os.makedirs(debug_dir, exist_ok=True)
            
            filename = f"step_{step_info.step:02d}_{step_info.screen_context}_{timestamp}.txt"
            filepath = os.path.join(debug_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"FLOW STEP ANALYSIS - Step {step_info.step}\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"SCREENSHOT: {screenshot_path}\n")
                f.write(f"TIMESTAMP: {timestamp}\n\n")
                
                f.write("STEP INFO:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Context: {step_info.screen_context}\n")
                f.write(f"Description: {step_info.description}\n")
                f.write(f"Target: {step_info.target_button}\n")
                f.write(f"Expected: {step_info.expected_buttons}\n")
                f.write(f"Action: {step_info.action}\n\n")
                
                f.write("UI ELEMENTS DETECTED:\n")
                f.write("-" * 40 + "\n")
                for i, element in enumerate(ui_elements):
                    content = element.get("content", "")
                    interactive = element.get("interactivity", False)
                    bbox = element.get("bbox", [])
                    f.write(f"{i+1:3d}. '{content}' ({'✓' if interactive else '✗'}) {bbox}\n")
                
                f.write(f"\nTotal Elements: {len(ui_elements)}\n")
                interactive_count = sum(1 for e in ui_elements if e.get("interactivity", False))
                f.write(f"Interactive Elements: {interactive_count}\n\n")
                
                f.write("QWEN ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Found: {qwen_analysis.get('found', False)}\n")
                f.write(f"Element Name: {qwen_analysis.get('element_name', 'None')}\n")
                f.write(f"Confidence: {qwen_analysis.get('confidence', 0.0)}\n")
                f.write(f"Reasoning: {qwen_analysis.get('reasoning', 'None')}\n")
                if 'raw_response' in qwen_analysis:
                    f.write(f"\nRaw Response:\n{qwen_analysis['raw_response']}\n")
                
                f.write("\n" + "="*80 + "\n")
            
            logger.info(f"Step analysis saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save step analysis: {e}")
    
    def _analyze_step_context(self, ui_analyzer, screenshot_path: str, 
                        step_info: FlowStep, directories: Dict = None) -> Dict[str, Any]:
        """Analyze UI with step-specific context and enhanced logging."""
        try:
            # Get UI elements from OmniParser
            output_dir = directories.get("omniparser_outputs") if directories else None
            ui_elements, labeled_image = ui_analyzer.omniparser.parse_screenshot(
                screenshot_path, save_output=True, output_dir=output_dir
            )
            
            logger.info(f"OmniParser detected {len(ui_elements)} UI elements")
            
            # Enhanced element logging
            if ui_elements:
                logger.info("Detected UI elements:")
                for i, element in enumerate(ui_elements[:15]):  # Log first 15
                    content = element.get("content", "")
                    interactive = element.get("interactivity", False)
                    bbox = element.get("bbox", [])
                    logger.info(f"  {i+1:2d}. '{content}' ({'Interactive' if interactive else 'Static'}) {bbox}")
            
            # Use Qwen with flow context and enhanced logging
            qwen_analysis = ui_analyzer.qwen.analyze_with_flow_context(ui_elements, step_info)
            
            # Save comprehensive analysis to debug file
            self._save_step_analysis_debug(step_info, ui_elements, qwen_analysis, screenshot_path)
            
            # Find target element based on Qwen analysis and step info
            target_element = None
            found_elements = []
            
            # Collect all element names for logging
            for element in ui_elements:
                element_name = element.get("content", "").strip()
                if element_name:
                    found_elements.append(element_name)
            
            # Look for target element
            if qwen_analysis["found"] and qwen_analysis["element_name"]:
                target_element = self._find_element_by_name(ui_elements, qwen_analysis["element_name"])
                if target_element:
                    logger.info(f"✅ Found target element via Qwen: {qwen_analysis['element_name']}")
                else:
                    logger.warning(f"⚠ Qwen found element '{qwen_analysis['element_name']}' but couldn't locate it")
            
            # Fallback: try to find element by expected buttons
            if not target_element:
                target_element = self._find_target_by_expected_buttons(ui_elements, step_info)
                if target_element:
                    logger.info(f"✅ Found target element via fallback: {target_element.get('content', 'Unknown')}")
            
            return {
                "ui_elements": ui_elements,
                "target_element": target_element,
                "found_elements": found_elements,
                "qwen_analysis": qwen_analysis,
                "labeled_image": labeled_image
            }
            
        except Exception as e:
            logger.error(f"Error in step context analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "ui_elements": [],
                "target_element": None,
                "found_elements": [],
                "qwen_analysis": {"found": False, "element_name": None, "confidence": 0.0},
                "labeled_image": None
            }
    
    def _find_element_by_name(self, ui_elements: List[Dict], element_name: str) -> Optional[Dict]:
        """Find UI element by name with fuzzy matching.
        
        Args:
            ui_elements: List of UI elements
            element_name: Name to search for
            
        Returns:
            Matching element or None
        """
        if not element_name:
            return None
            
        element_name_clean = element_name.strip().upper()
        
        # Try exact match first
        for element in ui_elements:
            content = element.get("content", "").strip().upper()
            if content == element_name_clean:
                logger.info(f"Found exact match: '{content}'")
                return element
        
        # Try contains match
        for element in ui_elements:
            content = element.get("content", "").strip().upper()
            if element_name_clean in content or content in element_name_clean:
                logger.info(f"Found contains match: '{content}' matches '{element_name_clean}'")
                return element
        
        # Try word-based matching
        element_words = element_name_clean.split()
        for element in ui_elements:
            content = element.get("content", "").strip().upper()
            content_words = content.split()
            
            # Check if any words match
            if any(word in content_words for word in element_words):
                logger.info(f"Found word match: '{content}' matches words from '{element_name_clean}'")
                return element
        
        logger.warning(f"No element found matching: '{element_name}'")
        return None
    
    def _find_target_by_expected_buttons(self, ui_elements: List[Dict], step_info: FlowStep) -> Optional[Dict]:
        """Find target element using expected buttons as fallback.
        
        Args:
            ui_elements: List of UI elements
            step_info: Current step information
            
        Returns:
            Target element or None
        """
        # Try target button first
        target = self._find_element_by_name(ui_elements, step_info.target_button)
        if target:
            logger.info(f"Found target via target_button: {step_info.target_button}")
            return target
        
        # Try expected buttons
        for expected_button in step_info.expected_buttons:
            target = self._find_element_by_name(ui_elements, expected_button)
            if target:
                logger.info(f"Found target via expected_button: {expected_button}")
                return target
        
        # Try interactive elements that contain keywords
        target_keywords = [step_info.target_button] + step_info.expected_buttons
        for element in ui_elements:
            if element.get("interactivity", False):
                content = element.get("content", "").upper()
                for keyword in target_keywords:
                    if keyword.upper() in content:
                        logger.info(f"Found interactive element with keyword '{keyword}': {content}")
                        return element
        
        return None
    
    def _create_context_prompt(self, step_info: FlowStep, ui_elements: List[Dict]) -> str:
        """Create a context-aware prompt for the current step.
        
        Args:
            step_info: Current step information
            ui_elements: Detected UI elements
            
        Returns:
            Context-specific prompt string
        """
        elements_text = []
        for i, element in enumerate(ui_elements):
            content = element.get("content", "Unknown")
            interactive = element.get("interactivity", False)
            elements_text.append(f"{i+1}. \"{content}\" ({'Interactive' if interactive else 'Static'})")
        
        elements_str = "\n".join(elements_text) if elements_text else "No UI elements detected"
        
        prompt = f"""
CONTEXT: {step_info.screen_context}
GOAL: {step_info.description}
TARGET BUTTON: {step_info.target_button}

UI ELEMENTS:
{elements_str}

EXPECTED BUTTONS: {', '.join(step_info.expected_buttons)}

Find the "{step_info.target_button}" button and respond:
FOUND: YES/NO
ELEMENT: [exact element name if found]
"""
        
        return prompt
    
    def _execute_step_action(self, step_info: FlowStep, analysis_result: Dict,
                           input_controller, screen_width: int, screen_height: int) -> bool:
        """Execute the action for the current step.
        
        Args:
            step_info: Current step information
            analysis_result: Result from UI analysis
            input_controller: Input controller instance
            screen_width: Screen width
            screen_height: Screen height
            
        Returns:
            True if action executed successfully
        """
        try:
            action = step_info.action.lower()
            
            if action == "click":
                target_element = analysis_result["target_element"]
                if target_element:
                    return input_controller.click_element(
                        target_element, screen_width, screen_height
                    )
                else:
                    logger.warning(f"Target button '{step_info.target_button}' not found")
                    return False
                    
            elif action == "click_anywhere":
                # Click in the center of the screen
                center_x = screen_width // 2
                center_y = screen_height // 2
                return input_controller.click_at_coordinates(center_x, center_y)
                
            elif action == "press_escape":
                return input_controller.press_key("escape")
                
            elif action == "press_enter":
                return input_controller.press_key("enter")
                
            else:
                logger.error(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return False
    
    def _verify_step_success(self, ui_analyzer, screenshot_manager, 
                           step_info: FlowStep) -> bool:
        """Verify that the step was successful by checking for success indicators.
        
        Args:
            ui_analyzer: UI analyzer instance
            screenshot_manager: Screenshot manager instance
            step_info: Current step information
            
        Returns:
            True if success indicators found
        """
        try:
            # Take a new screenshot to check results
            verification_screenshot = screenshot_manager.take_screenshot()
            if not verification_screenshot:
                return False
            
            # Get UI elements
            ui_elements, _ = ui_analyzer.omniparser.parse_screenshot(
                verification_screenshot, save_output=False
            )
            
            # Check for success indicators
            found_indicators = []
            for element in ui_elements:
                element_name = element.get("content", "").upper()
                for indicator in step_info.success_indicators:
                    if indicator.upper() in element_name:
                        found_indicators.append(indicator)
                        logger.info(f"Found success indicator: {indicator}")
            
            return len(found_indicators) > 0
            
        except Exception as e:
            logger.error(f"Error verifying step success: {e}")
            return False
    
    def advance_step(self) -> bool:
        """Advance to the next step in the flow.
        
        Returns:
            True if advanced successfully, False if no more steps
        """
        if not self.current_game or self.current_game not in self.flows:
            return False
        
        game_flow = self.flows[self.current_game]['flow']
        if self.current_step < len(game_flow) - 1:
            self.current_step += 1
            logger.info(f"Advanced to step {self.current_step + 1}")
            return True
        else:
            logger.info("Reached end of flow")
            return False
    
    def is_flow_complete(self) -> bool:
        """Check if the current flow is complete.
        
        Returns:
            True if all steps are completed
        """
        if not self.current_game or self.current_game not in self.flows:
            return True
        
        game_flow = self.flows[self.current_game]['flow']
        return self.current_step >= len(game_flow)
    
    def detect_benchmark_results(self, ui_analyzer, screenshot_manager) -> bool:
        """Detect if benchmark results are displayed using flow configuration.
        
        Args:
            ui_analyzer: UI analyzer instance
            screenshot_manager: Screenshot manager instance
            
        Returns:
            True if benchmark results detected
        """
        try:
            screenshot_path = screenshot_manager.take_screenshot()
            if not screenshot_path:
                return False
            
            ui_elements, _ = ui_analyzer.omniparser.parse_screenshot(
                screenshot_path, save_output=False
            )
            
            # Check for FPS indicators
            fps_indicators = self.result_detection.get('fps_indicators', [])
            completion_indicators = self.result_detection.get('completion_indicators', [])
            
            found_fps = 0
            found_completion = 0
            
            for element in ui_elements:
                element_name = element.get("content", "").upper()
                
                for indicator in fps_indicators:
                    if indicator.upper() in element_name:
                        found_fps += 1
                        
                for indicator in completion_indicators:
                    if indicator.upper() in element_name:
                        found_completion += 1
            
            # Results detected if we find FPS metrics OR completion indicators
            results_found = found_fps >= 2 or found_completion >= 1
            
            if results_found:
                logger.info(f"Benchmark results detected! FPS indicators: {found_fps}, Completion: {found_completion}")
            
            return results_found
            
        except Exception as e:
            logger.error(f"Error detecting benchmark results: {e}")
            return False
    
    def get_exit_flow(self) -> List[FlowStep]:
        """Get the exit flow steps.
        
        Returns:
            List of exit flow steps
        """
        try:
            exit_steps = []
            common_exit = self.exit_flow.get('common', {}).get('flow', [])
            
            for step_data in common_exit:
                exit_steps.append(FlowStep(
                    step=step_data['step'],
                    screen_context=step_data['screen_context'],
                    description=step_data['description'],
                    expected_buttons=step_data.get('expected_buttons', []),
                    target_button=step_data.get('target_button', ''),
                    action=step_data['action'],
                    max_retries=step_data['max_retries'],
                    success_indicators=step_data['success_indicators']
                ))
            
            return exit_steps
            
        except Exception as e:
            logger.error(f"Error getting exit flow: {e}")
            return []
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get a summary of the current flow execution.
        
        Returns:
            Flow execution summary
        """
        if not self.current_game:
            return {"error": "No current game set"}
        
        total_steps = len(self.flows[self.current_game]['flow'])
        completed_steps = len([h for h in self.step_history if h.get('success', False)])
        
        return {
            "game": self.current_game,
            "total_steps": total_steps,
            "current_step": self.current_step + 1,
            "completed_steps": completed_steps,
            "step_history": self.step_history,
            "is_complete": self.is_flow_complete()
        }