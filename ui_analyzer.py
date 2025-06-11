"""
UI Analyzer module - FIXED VERSION that saves debug outputs for EVERY analysis.
"""
import os
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import win32api

from omniparser import OmniParserV2
from qwen import QwenDecisionMaker

logger = logging.getLogger("UIAnalyzer")

class UIAnalyzer:
    """Fixed UI Analyzer that saves debug outputs for every screenshot analysis."""
    
    def __init__(self, omniparser_config: Dict, qwen_config: Dict, debug_config: Dict):
        """Initialize the UI analyzer."""
        self.debug_config = debug_config
        self.omniparser = OmniParserV2(omniparser_config)
        self.qwen = QwenDecisionMaker(qwen_config)
        
        # Counter for unique debug file naming
        self.analysis_counter = 0
        
        # Get screen dimensions for coordinate conversion
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        
        logger.info("UI Analyzer initialized")
    
    def analyze_screenshot(self, screenshot_path: str, directories: Dict, context: str = "main_menu") -> Dict[str, Any]:
        """Analyze screenshot with debug output saving for EVERY analysis."""
        self.analysis_counter += 1
        analysis_start_time = time.time()
        
        logger.info(f"Starting analysis #{self.analysis_counter} for context: {context}")
        
        # Parse UI elements with forced debug output
        ui_elements, labeled_image_path = self.omniparser.parse_screenshot(
            screenshot_path, 
            save_output=True,  # ALWAYS save output
            output_dir=directories.get("analyzed_screenshots")
        )
        
        # FIXED: Save OmniParser debug data for EVERY analysis
        if self.debug_config.get("save_omniparser_outputs", True):
            self._save_omniparser_output(ui_elements, screenshot_path, directories, context, self.analysis_counter)
        
        # Make decision using Qwen with debug saving
        decision = self._make_decision_with_debug(ui_elements, context, directories, self.analysis_counter)
        
        # Find target coordinates
        target_coordinates = None
        if decision.get("action") == "CLICK" and decision.get("target"):
            target_coordinates = self._find_target_coordinates(ui_elements, decision["target"])
        
        # Compile analysis result
        analysis_result = {
            "timestamp": time.time(),
            "context": context,
            "analysis_id": self.analysis_counter,
            "ui_elements": ui_elements,
            "interactive_elements": self._get_interactive_elements(ui_elements),
            "decision": decision,
            "target_coordinates": target_coordinates,
            "labeled_image_path": labeled_image_path,
            "processing_time": time.time() - analysis_start_time
        }
        
        # FIXED: Save analysis summary for EVERY analysis
        if self.debug_config.get("save_omniparser_outputs", True):
            self._save_analysis_summary(analysis_result, directories, self.analysis_counter)
        
        logger.info(f"Analysis #{self.analysis_counter} completed in {analysis_result['processing_time']:.2f}s")
        logger.info(f"Found {len(ui_elements)} elements, {len(analysis_result['interactive_elements'])} interactive")
        
        return analysis_result
    
    def _save_omniparser_output(self, ui_elements: List[Dict], screenshot_path: str, 
                               directories: Dict, context: str, analysis_id: int):
        """Save OmniParser output with unique naming for each analysis."""
        try:
            omniparser_dir = directories.get("omniparser_output")
            if not omniparser_dir:
                return
            
            os.makedirs(omniparser_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"omniparser_{context}_analysis_{analysis_id:03d}_{timestamp}.txt"
            filepath = os.path.join(omniparser_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== OMNIPARSER OUTPUT ===\n")
                f.write(f"Analysis ID: {analysis_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Context: {context}\n")
                f.write(f"Screenshot: {screenshot_path}\n")
                f.write(f"Total Elements: {len(ui_elements)}\n")
                f.write(f"\n=== UI ELEMENTS ===\n")
                
                interactive_count = 0
                for i, element in enumerate(ui_elements):
                    content = element.get('content', 'Unknown')
                    is_interactive = element.get('interactivity', False)
                    element_type = element.get('element_type', 'unknown')
                    bbox = element.get('bbox', [])
                    
                    if is_interactive:
                        interactive_count += 1
                    
                    f.write(f"\n{i+1:2d}. '{content}'\n")
                    f.write(f"    Type: {element_type}\n")
                    f.write(f"    Interactive: {is_interactive}\n")
                    f.write(f"    BBox: {bbox}\n")
                
                f.write(f"\n=== SUMMARY ===\n")
                f.write(f"Total Elements: {len(ui_elements)}\n")
                f.write(f"Interactive Elements: {interactive_count}\n")
                f.write(f"Static Elements: {len(ui_elements) - interactive_count}\n")
                f.write(f"\n=== END OUTPUT ===\n")
            
            logger.debug(f"OmniParser output saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save OmniParser output: {e}")
    
    def _make_decision_with_debug(self, ui_elements: List[Dict], context: str, directories: Dict, analysis_id: int) -> Dict[str, Any]:
        """Make decision using Qwen with debug saving for each analysis."""
        try:
            # Create decision prompt
            elements_text = self._format_ui_elements_for_prompt(ui_elements)
            prompt = self._create_context_prompt(context, elements_text)
            
            # FIXED: Save prompt with unique naming for each analysis
            if self.debug_config.get("save_qwen_responses", True):
                self._save_qwen_prompt(prompt, context, directories, analysis_id)
            
            # Generate response with Qwen
            response = self.qwen._generate_response(prompt)
            
            # FIXED: Save response with unique naming for each analysis
            if self.debug_config.get("save_qwen_responses", True):
                self._save_qwen_response(response, context, directories, analysis_id)
            
            # Parse the response
            decision = self._parse_decision_response(response)
            
            logger.debug(f"Decision for analysis {analysis_id}: {decision['action']} -> {decision.get('target', 'N/A')}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in decision making for analysis {analysis_id}: {e}")
            return {
                "action": "WAIT",
                "target": None,
                "confidence": 0.1,
                "reasoning": f"Error: {str(e)}"
            }
    
    def _save_qwen_prompt(self, prompt: str, context: str, directories: Dict, analysis_id: int):
        """Save Qwen prompt with unique naming for each analysis."""
        try:
            prompt_dir = directories.get("qwen_prompt")
            if not prompt_dir:
                return
            
            os.makedirs(prompt_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_prompt_{context}_analysis_{analysis_id:03d}_{timestamp}.txt"
            filepath = os.path.join(prompt_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== QWEN PROMPT ===\n")
                f.write(f"Analysis ID: {analysis_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Context: {context}\n")
                f.write(f"\n=== PROMPT CONTENT ===\n")
                f.write(prompt)
                f.write(f"\n=== END PROMPT ===\n")
            
            logger.debug(f"Qwen prompt saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save Qwen prompt: {e}")
    
    def _save_qwen_response(self, response: str, context: str, directories: Dict, analysis_id: int):
        """Save Qwen response with unique naming for each analysis."""
        try:
            response_dir = directories.get("qwen_responses")
            if not response_dir:
                return
            
            os.makedirs(response_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_response_{context}_analysis_{analysis_id:03d}_{timestamp}.txt"
            filepath = os.path.join(response_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== QWEN RESPONSE ===\n")
                f.write(f"Analysis ID: {analysis_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Context: {context}\n")
                f.write(f"\n=== RESPONSE CONTENT ===\n")
                f.write(response)
                f.write(f"\n=== END RESPONSE ===\n")
            
            logger.debug(f"Qwen response saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save Qwen response: {e}")
    
    def _save_analysis_summary(self, analysis_result: Dict, directories: Dict, analysis_id: int):
        """Save analysis summary with unique naming for each analysis."""
        try:
            runtime_logs_dir = directories.get("runtime_logs")
            if not runtime_logs_dir:
                return
            
            os.makedirs(runtime_logs_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_summary_{analysis_id:03d}_{timestamp}.txt"
            filepath = os.path.join(runtime_logs_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== ANALYSIS SUMMARY ===\n")
                f.write(f"Analysis ID: {analysis_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Context: {analysis_result['context']}\n")
                f.write(f"Processing Time: {analysis_result['processing_time']:.2f}s\n")
                f.write(f"Total Elements: {len(analysis_result['ui_elements'])}\n")
                f.write(f"Interactive Elements: {len(analysis_result['interactive_elements'])}\n")
                f.write(f"Labeled Image: {analysis_result['labeled_image_path']}\n")
                
                decision = analysis_result['decision']
                f.write(f"\n=== DECISION ===\n")
                f.write(f"Action: {decision['action']}\n")
                f.write(f"Target: {decision.get('target', 'None')}\n")
                f.write(f"Confidence: {decision['confidence']:.2f}\n")
                f.write(f"Reasoning: {decision.get('reasoning', 'None')}\n")
                
                # List all interactive elements for debugging
                f.write(f"\n=== INTERACTIVE ELEMENTS ===\n")
                for i, element in enumerate(analysis_result['interactive_elements']):
                    content = element.get('content', 'Unknown')
                    bbox = element.get('bbox', [])
                    f.write(f"{i+1:2d}. '{content}' - {bbox}\n")
                
                f.write(f"\n=== END SUMMARY ===\n")
            
            logger.debug(f"Analysis summary saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis summary: {e}")
    
    def _save_analysis_debug(self, ui_elements: List[Dict], screenshot_path: str, directories: Dict, context: str):
        """Additional method to force save debug info - called from adaptive executor."""
        self._save_omniparser_output(ui_elements, screenshot_path, directories, context, self.analysis_counter)
    
    def _format_ui_elements_for_prompt(self, ui_elements: List[Dict]) -> str:
        """Format UI elements for inclusion in prompts with better structure."""
        if not ui_elements:
            return "No UI elements detected"
        
        elements = []
        interactive_count = 0
        
        for i, element in enumerate(ui_elements[:15]):  # Limit to first 15 elements
            content = element.get("content", "").strip()
            if content:
                interactive = element.get("interactivity", False)
                bbox = element.get("bbox", [])
                
                if interactive:
                    interactive_count += 1
                    elements.append(f"- '{content}' [CLICKABLE] {bbox}")
                else:
                    elements.append(f"- '{content}' [TEXT] {bbox}")
        
        if not elements:
            return "No readable elements found"
        
        header = f"Found {len(elements)} elements ({interactive_count} clickable):\n"
        return header + "\n".join(elements)
    
    def _create_context_prompt(self, context: str, elements_text: str) -> str:
        """Create context-specific prompt for Qwen with enhanced confirmation dialog handling."""
        base_prompt = f"""
CONTEXT: {context}
GOAL: Navigate through game menus to find and start benchmark

UI ELEMENTS DETECTED:
{elements_text}

Based on the detected UI elements, what should we do next?
"""
        
        if context == "main_menu":
            return base_prompt + """
We're at the main menu. Look for benchmark button or options/settings/graphics menu to access benchmark.
Respond with:
ACTION: CLICK/WAIT/BACK
TARGET: [element name if clicking]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""
        elif context in ["options", "settings", "navigation"]:
            return base_prompt + """
We're in a settings/options menu. Look for graphics/video settings or benchmark option.
Respond with:
ACTION: CLICK/WAIT/BACK
TARGET: [element name if clicking]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""
        elif "confirm" in context.lower() or any("confirm" in str(elem.get("content", "")).lower() for elem in []):
            return base_prompt + """
We're in a confirmation dialog. Look for CONFIRM, YES, OK, START buttons to proceed.
AVOID clicking on CANCEL, NO, BACK buttons.
Focus on buttons that will START or CONFIRM the benchmark.
Respond with:
ACTION: CLICK/WAIT/BACK
TARGET: [element name if clicking]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""
        else:
            return base_prompt + """
Navigate to find the benchmark option.
Respond with:
ACTION: CLICK/WAIT/BACK
TARGET: [element name if clicking]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""
    
    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse a decision response from Qwen with better error handling."""
        try:
            decision = {
                "action": "WAIT",
                "target": None,
                "confidence": 0.5,
                "reasoning": "Default decision"
            }
            
            lines = response.split('\n')
            for line in lines:
                line_upper = line.strip().upper()
                if line_upper.startswith("ACTION:"):
                    action_part = line_upper.split(":", 1)[1].strip()
                    if any(a in action_part for a in ["CLICK", "WAIT", "BACK"]):
                        for action in ["CLICK", "WAIT", "BACK"]:
                            if action in action_part:
                                decision["action"] = action
                                break
                elif line_upper.startswith("TARGET:"):
                    target = line.split(":", 1)[1].strip()
                    if target and target.upper() not in ["N/A", "NONE", "NULL"]:
                        decision["target"] = target
                elif line_upper.startswith("CONFIDENCE:"):
                    try:
                        conf_str = line.split(":", 1)[1].strip()
                        import re
                        numbers = re.findall(r'[\d.]+', conf_str)
                        if numbers:
                            confidence = float(numbers[0])
                            if confidence > 1.0:
                                confidence = confidence / 100.0
                            decision["confidence"] = min(max(confidence, 0.0), 1.0)
                    except (ValueError, IndexError):
                        pass
                elif line_upper.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                    if reasoning:
                        decision["reasoning"] = reasoning
            
            return decision
            
        except Exception as e:
            logger.error(f"Error parsing decision: {e}")
            return {
                "action": "WAIT",
                "target": None,
                "confidence": 0.3,
                "reasoning": f"Parse error: {str(e)}"
            }
    
    def _find_target_coordinates(self, ui_elements: List[Dict], target_name: str) -> Optional[Tuple[int, int]]:
        """Find screen coordinates for a target element."""
        try:
            target_element = self._find_element_by_name(ui_elements, target_name)
            if target_element:
                center_x, center_y = self.omniparser.get_element_center(
                    target_element, self.screen_width, self.screen_height
                )
                return center_x, center_y
            return None
        except Exception as e:
            logger.error(f"Error finding target coordinates: {e}")
            return None
    
    def _find_element_by_name(self, ui_elements: List[Dict], element_name: str) -> Optional[Dict]:
        """Find a UI element by its name/content with better matching."""
        element_name_lower = element_name.lower()
        
        # First try exact match
        for element in ui_elements:
            content = element.get("content", "").lower()
            if content == element_name_lower:
                return element
        
        # Then try partial match
        for element in ui_elements:
            content = element.get("content", "").lower()
            if element_name_lower in content or content in element_name_lower:
                return element
        
        # Finally try keyword match
        keywords = element_name_lower.split()
        for element in ui_elements:
            content = element.get("content", "").lower()
            if any(keyword in content for keyword in keywords):
                return element
        
        return None
    
    def _get_interactive_elements(self, ui_elements: List[Dict]) -> List[Dict]:
        """Get all interactive elements from the UI elements list."""
        return [elem for elem in ui_elements if elem.get("interactivity", False)]
    
    def detect_benchmark_option(self, screenshot_path: str, directories: Dict) -> Dict[str, Any]:
        """Detect benchmark options with debug output."""
        try:
            # Use the main analyze_screenshot method to ensure debug output is saved
            analysis_result = self.analyze_screenshot(screenshot_path, directories, "benchmark_detection")
            ui_elements = analysis_result["ui_elements"]
            
            if not ui_elements:
                return {"found": False, "element": None, "action": "WAIT"}
            
            # Find benchmark-related elements
            benchmark_keywords = ["benchmark", "test", "performance"]
            benchmark_elements = self.find_elements_by_keywords(
                ui_elements, benchmark_keywords, interactive_only=True
            )
            
            target_element = benchmark_elements[0] if benchmark_elements else None
            
            return {
                "found": target_element is not None,
                "element": target_element,
                "action": "CLICK" if target_element else "WAIT",
                "target": target_element.get("content") if target_element else None,
                "confidence": 0.9 if target_element else 0.1
            }
            
        except Exception as e:
            logger.error(f"Error in benchmark detection: {e}")
            return {"found": False, "element": None, "action": "WAIT"}
    
    def detect_benchmark_results(self, screenshot_path: str, directories: Dict) -> Dict[str, Any]:
        """Detect benchmark results with debug output."""
        try:
            # Use the main analyze_screenshot method to ensure debug output is saved
            analysis_result = self.analyze_screenshot(screenshot_path, directories, "result_detection")
            ui_elements = analysis_result["ui_elements"]
            
            if not ui_elements:
                return {"found": False}
            
            # Use Qwen for result detection
            found = self.qwen.detect_benchmark_results(ui_elements)
            
            # Also check for specific result keywords
            result_keywords = ["fps", "average", "minimum", "maximum", "score", "results", "finished"]
            result_elements = self.find_elements_by_keywords(ui_elements, result_keywords)
            
            keyword_found = len(result_elements) >= 2
            
            return {
                "found": found or keyword_found,
                "qwen_detection": found,
                "keyword_detection": keyword_found,
                "result_elements": len(result_elements),
                "details": [e.get("content", "") for e in result_elements[:5]]
            }
            
        except Exception as e:
            logger.error(f"Error in result detection: {e}")
            return {"found": False}
    
    def find_elements_by_keywords(self, ui_elements: List[Dict], keywords: List[str], interactive_only: bool = True) -> List[Dict]:
        """Find UI elements containing specific keywords."""
        matching_elements = self.omniparser.find_elements_by_keyword(ui_elements, keywords)
        
        if interactive_only:
            matching_elements = [elem for elem in matching_elements if elem.get("interactivity", False)]
        
        return matching_elements
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.omniparser.cleanup()
            self.qwen.cleanup()
            logger.info("UI Analyzer resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")