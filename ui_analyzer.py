"""
UI Analyzer module that combines OmniParser V2 and Qwen 2.5 for comprehensive UI analysis.
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
    """Combines OmniParser V2 and Qwen 2.5 for intelligent UI analysis and decision making."""
    
    def __init__(self, omniparser_config: Dict, qwen_config: Dict, debug_config: Dict):
        """Initialize the UI analyzer with enhanced debug logging."""
        self.debug_config = debug_config
        self.omniparser = OmniParserV2(omniparser_config)
        self.qwen = QwenDecisionMaker(qwen_config)
        
        # Get screen dimensions for coordinate conversion
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        
        # Log debug configuration
        logger.info(f"Debug config: {debug_config}")
        logger.info(f"Save Qwen responses: {debug_config.get('save_qwen_responses', False)}")
        logger.info(f"Save OmniParser outputs: {debug_config.get('save_omniparser_outputs', False)}")
        
        logger.info("UI Analyzer initialized successfully")
    
    def analyze_screenshot(self, screenshot_path: str, directories: Dict, context: str = "main_menu") -> Dict[str, Any]:
        """Analyze a screenshot and provide decision recommendations.
        
        Args:
            screenshot_path: Path to the screenshot
            directories: Directory structure for saving outputs
            context: Current context (e.g., 'main_menu', 'options', 'benchmark')
            
        Returns:
            Complete analysis result with UI elements and decision
        """
        try:
            logger.info(f"Analyzing screenshot: {screenshot_path}")
            start_time = time.time()
            
            # Step 1: Parse UI elements with OmniParser V2
            ui_elements, labeled_image_path = self._parse_ui_elements(
                screenshot_path, directories.get("omniparser_outputs")
            )
            
            if not ui_elements:
                logger.warning("No UI elements detected")
                return self._create_empty_analysis_result()
            
            # Step 2: Make decision with Qwen 2.5 based on context
            decision = self._make_decision(ui_elements, context, directories.get("qwen_responses"))
            
            # Step 3: Find target element coordinates if needed
            target_coordinates = None
            if decision.get("action") == "CLICK" and decision.get("target"):
                target_coordinates = self._find_target_coordinates(ui_elements, decision["target"])
            
            # Step 4: Compile analysis result
            analysis_result = {
                "timestamp": time.time(),
                "context": context,
                "ui_elements": ui_elements,
                "interactive_elements": self._get_interactive_elements(ui_elements),
                "decision": decision,
                "target_coordinates": target_coordinates,
                "labeled_image_path": labeled_image_path,
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Analysis completed in {analysis_result['processing_time']:.2f}s")
            logger.info(f"Found {len(ui_elements)} elements, {len(analysis_result['interactive_elements'])} interactive")
            logger.info(f"Decision: {decision['action']} -> {decision.get('target', 'N/A')} (confidence: {decision['confidence']:.2f})")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error during screenshot analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_empty_analysis_result()
    
    def detect_benchmark_option(self, screenshot_path: str, directories: Dict) -> Dict[str, Any]:
        """Specifically detect benchmark options in the current screen.
        
        Args:
            screenshot_path: Path to the screenshot
            directories: Directory structure for saving outputs
            
        Returns:
            Benchmark detection result with element if found
        """
        try:
            logger.info("Detecting benchmark options...")
            
            # Parse UI elements
            ui_elements, _ = self._parse_ui_elements(screenshot_path, directories.get("omniparser_outputs"))
            
            if not ui_elements:
                return {"found": False, "element": None, "action": "WAIT"}
            
            # Use simplified Qwen method
            decision = self.qwen.find_benchmark_button(ui_elements)
            
            # Find the actual element if action is CLICK
            target_element = None
            if decision["action"] == "CLICK" and decision["target"]:
                target_element = self._find_element_by_name(ui_elements, decision["target"])
            
            result = {
                "found": decision["action"] == "CLICK" and target_element is not None,
                "element": target_element,
                "action": decision["action"],
                "target": decision["target"],
                "confidence": decision["confidence"]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in benchmark detection: {e}")
            return {"found": False, "element": None, "action": "WAIT"}
    
    def detect_benchmark_results(self, screenshot_path: str, directories: Dict) -> Dict[str, Any]:
        """Detect if benchmark results are displayed in the current screen.
        
        Args:
            screenshot_path: Path to the screenshot
            directories: Directory structure for saving outputs
            
        Returns:
            Simple boolean result
        """
        try:
            logger.info("Detecting benchmark results...")
            
            # Parse UI elements
            ui_elements, _ = self._parse_ui_elements(screenshot_path, directories.get("omniparser_outputs"))
            
            if not ui_elements:
                return {"found": False}
            
            # Use simplified Qwen method
            found = self.qwen.detect_benchmark_results(ui_elements)
            
            return {"found": found}
            
        except Exception as e:
            logger.error(f"Error in result detection: {e}")
            return {"found": False}
    
    def find_exit_option(self, screenshot_path: str, directories: Dict) -> Dict[str, Any]:
        """Find exit/quit options in the current screen.
        
        Args:
            screenshot_path: Path to the screenshot
            directories: Directory structure for saving outputs
            
        Returns:
            Exit option result with element if found
        """
        try:
            logger.info("Finding exit options...")
            
            # Parse UI elements
            ui_elements, _ = self._parse_ui_elements(screenshot_path, directories.get("omniparser_outputs"))
            
            if not ui_elements:
                return {"found": False, "element": None}
            
            # Use simplified Qwen method
            exit_element_name = self.qwen.find_exit_button(ui_elements)
            
            # Find the actual element
            target_element = None
            if exit_element_name:
                target_element = self._find_element_by_name(ui_elements, exit_element_name)
            
            return {
                "found": target_element is not None,
                "element": target_element
            }
            
        except Exception as e:
            logger.error(f"Error finding exit option: {e}")
            return {"found": False, "element": None}
    
    def _parse_ui_elements(self, screenshot_path: str, output_dir: Optional[str]) -> Tuple[List[Dict], Optional[str]]:
        """Parse UI elements using OmniParser V2.
        
        Args:
            screenshot_path: Path to the screenshot
            output_dir: Directory to save OmniParser outputs
            
        Returns:
            Tuple of (ui_elements, labeled_image_path)
        """
        save_output = self.debug_config["save_omniparser_outputs"] and output_dir is not None
        return self.omniparser.parse_screenshot(screenshot_path, save_output, output_dir)
    
    def _make_decision(self, ui_elements: List[Dict], context: str, responses_dir: Optional[str]) -> Dict[str, Any]:
        """Make decision using Qwen 2.5 based on UI elements and context.
        
        Args:
            ui_elements: Parsed UI elements
            context: Current context
            responses_dir: Directory to save Qwen responses
            
        Returns:
            Decision dictionary
        """
        try:
            if context == "main_menu":
                decision = self.qwen.analyze_main_menu(ui_elements)
            elif context in ["options", "settings", "navigation"]:
                decision = self.qwen.analyze_navigation(ui_elements, context)
            else:
                # Default to main menu analysis
                decision = self.qwen.analyze_main_menu(ui_elements)
            
            # Save debug info if enabled
            if self.debug_config["save_qwen_responses"] and responses_dir:
                self.qwen.save_response(
                    f"Decision making for context: {context}",
                    str(decision),
                    responses_dir,
                    f"decision_{context}"
                )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            return {
                "action": "WAIT",
                "target": None,
                "confidence": 0.3,
                "reasoning": "Error in decision making"
            }
    
    def _find_target_coordinates(self, ui_elements: List[Dict], target_name: str) -> Optional[Tuple[int, int]]:
        """Find screen coordinates for a target element.
        
        Args:
            ui_elements: List of UI elements
            target_name: Name of the target element
            
        Returns:
            Screen coordinates (x, y) or None if not found
        """
        try:
            target_element = self._find_element_by_name(ui_elements, target_name)
            if target_element:
                center_x, center_y = self.omniparser.get_element_center(
                    target_element, self.screen_width, self.screen_height
                )
                logger.info(f"Target '{target_name}' found at coordinates ({center_x}, {center_y})")
                return center_x, center_y
            else:
                logger.warning(f"Target element '{target_name}' not found")
                return None
                
        except Exception as e:
            logger.error(f"Error finding target coordinates: {e}")
            return None
    
    def _find_element_by_name(self, ui_elements: List[Dict], element_name: str) -> Optional[Dict]:
        """Find a UI element by its name/content.
        
        Args:
            ui_elements: List of UI elements
            element_name: Name to search for
            
        Returns:
            Matching element or None
        """
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
        """Get all interactive elements from the UI elements list.
        
        Args:
            ui_elements: List of UI elements
            
        Returns:
            List of interactive elements
        """
        return [elem for elem in ui_elements if elem.get("interactivity", False)]
    
    def _create_empty_analysis_result(self) -> Dict[str, Any]:
        """Create an empty analysis result for error cases.
        
        Returns:
            Empty analysis result dictionary
        """
        return {
            "timestamp": time.time(),
            "context": "unknown",
            "ui_elements": [],
            "interactive_elements": [],
            "decision": {
                "action": "WAIT",
                "target": None,
                "confidence": 0.0,
                "reasoning": "Analysis failed"
            },
            "target_coordinates": None,
            "labeled_image_path": None,
            "processing_time": 0.0
        }
    
    def find_elements_by_keywords(self, ui_elements: List[Dict], keywords: List[str], interactive_only: bool = True) -> List[Dict]:
        """Find UI elements containing specific keywords.
        
        Args:
            ui_elements: List of UI elements
            keywords: Keywords to search for
            interactive_only: Whether to only return interactive elements
            
        Returns:
            List of matching elements
        """
        matching_elements = self.omniparser.find_elements_by_keyword(ui_elements, keywords)
        
        if interactive_only:
            matching_elements = [elem for elem in matching_elements if elem.get("interactivity", False)]
        
        return matching_elements
    
    def get_best_clickable_element(self, ui_elements: List[Dict], preferred_keywords: List[str] = None) -> Optional[Dict]:
        """Get the best clickable element based on preferences.
        
        Args:
            ui_elements: List of UI elements
            preferred_keywords: Preferred keywords for element selection
            
        Returns:
            Best clickable element or None
        """
        interactive_elements = self._get_interactive_elements(ui_elements)
        
        if not interactive_elements:
            return None
        
        # If preferred keywords are provided, try to find matching elements
        if preferred_keywords:
            preferred_elements = self.find_elements_by_keywords(interactive_elements, preferred_keywords, True)
            if preferred_elements:
                return preferred_elements[0]  # Return first matching element
        
        # Otherwise return the first interactive element
        return interactive_elements[0]
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.omniparser.cleanup()
            self.qwen.cleanup()
            logger.info("UI Analyzer resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during UI Analyzer cleanup: {e}")