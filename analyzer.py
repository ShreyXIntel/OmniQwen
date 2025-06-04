"""
UI Element Analyzer Module
Handles analysis of parsed content and finding specific UI elements
"""

import logging
from config import Config
from input_controller import InputController


class UIAnalyzer:
    def __init__(self, input_controller=None):
        self.logger = logging.getLogger(Config.LOGGER_NAME)
        self.input_controller = input_controller or InputController()
    
    def has_element_containing(self, parsed_content, target_texts):
        """Check if the parsed content contains elements with any of the target texts."""
        if (
            not parsed_content
            or not isinstance(parsed_content, list)
            or len(parsed_content) == 0
        ):
            self.logger.warning(
                "Empty or invalid parsed content when checking for elements"
            )
            return False
        
        target_texts = [text.lower() for text in target_texts]
        
        for element in parsed_content:
            if not isinstance(element, dict):
                continue
            
            content = element.get("content", "").lower()
            for target in target_texts:
                if target in content:
                    self.logger.info(f"Found element containing '{target}': {content}")
                    return True
        
        return False
    
    def has_multiple_elements_containing(self, parsed_content, target_lists):
        """Check if all sets of target texts are found in the parsed content."""
        for target_texts in target_lists:
            if not self.has_element_containing(parsed_content, target_texts):
                return False
        return True
    
    def find_element(self, parsed_content, target_texts, require_interactivity=False):
        """Find an element containing one of the target texts."""
        if not parsed_content or not isinstance(parsed_content, list):
            self.logger.warning("Empty or invalid parsed content when finding element")
            return None
        
        target_texts = [text.lower() for text in target_texts]
        
        # First pass: look for interactive elements if required
        if require_interactivity:
            for element in parsed_content:
                if not isinstance(element, dict):
                    continue
                
                content = element.get("content", "").lower()
                for target in target_texts:
                    if target in content and element.get("interactivity", False):
                        self.logger.info(
                            f"Found interactive element matching '{target}': {content}"
                        )
                        return element
        
        # Second pass: look for any matching elements
        for element in parsed_content:
            if not isinstance(element, dict):
                continue
            
            content = element.get("content", "").lower()
            for target in target_texts:
                if target in content:
                    interactivity_status = "interactive" if element.get("interactivity", False) else "non-interactive"
                    self.logger.info(
                        f"Found {interactivity_status} element matching '{target}': {content}"
                    )
                    return element
        
        return None
    
    def find_and_click_element(self, parsed_content, target_texts, fallback_targets=None):
        """Find an element containing one of the target texts and click it."""
        # Try primary targets first (interactive elements preferred)
        element = self.find_element(parsed_content, target_texts, require_interactivity=True)
        if element:
            return self.input_controller.click_at_element(element)
        
        # Try primary targets without interactivity requirement
        element = self.find_element(parsed_content, target_texts, require_interactivity=False)
        if element:
            return self.input_controller.click_at_element(element)
        
        # Try fallback targets if provided
        if fallback_targets:
            element = self.find_element(parsed_content, fallback_targets, require_interactivity=True)
            if element:
                return self.input_controller.click_at_element(element)
            
            element = self.find_element(parsed_content, fallback_targets, require_interactivity=False)
            if element:
                return self.input_controller.click_at_element(element)
        
        self.logger.warning(
            f"No elements found matching any of these targets: {target_texts}"
        )
        return False
    
    def log_all_detected_elements(self, parsed_content, context="Current"):
        """Log all detected UI elements for debugging purposes."""
        if not parsed_content or not isinstance(parsed_content, list):
            self.logger.info(f"{context} detected UI elements: None")
            return
        
        self.logger.info(f"{context} detected UI elements:")
        for i, element in enumerate(parsed_content):
            if isinstance(element, dict) and 'content' in element:
                interactive = " (interactive)" if element.get('interactivity', False) else ""
                self.logger.info(f"  {i+1}. {element.get('content', '')}{interactive}")
    
    def check_for_main_menu(self, parsed_content):
        """Check if the current screen shows the main menu."""
        return self.has_multiple_elements_containing(
            parsed_content, 
            Config.MAIN_MENU_INDICATORS
        )
    
    def check_for_benchmark_results(self, parsed_content):
        """Check if the current screen shows benchmark results."""
        return self.has_element_containing(parsed_content, Config.FPS_INDICATORS)
    
    def find_options_menu(self, parsed_content):
        """Find and click on options/settings menu."""
        return self.find_and_click_element(
            parsed_content,
            Config.OPTIONS_TARGETS,
            Config.OPTIONS_FALLBACKS
        )
    
    def find_benchmark_option(self, parsed_content):
        """Find and click on benchmark option."""
        return self.find_and_click_element(
            parsed_content,
            Config.BENCHMARK_TARGETS,
            Config.BENCHMARK_FALLBACKS
        )
    
    def find_graphics_option(self, parsed_content):
        """Find and click on graphics/display option."""
        return self.find_and_click_element(
            parsed_content,
            Config.GRAPHICS_TARGETS,
            Config.GRAPHICS_FALLBACKS
        )
    
    def find_back_button(self, parsed_content):
        """Find and click on back/return button."""
        return self.find_and_click_element(parsed_content, Config.BACK_BUTTONS)
    
    def find_continue_button(self, parsed_content):
        """Find and click on continue/ok button."""
        return self.find_and_click_element(
            parsed_content,
            Config.CONTINUE_BUTTONS,
            Config.CONTINUE_FALLBACKS
        )
    
    def find_exit_option(self, parsed_content):
        """Find and click on exit/quit option."""
        return self.find_and_click_element(parsed_content, Config.EXIT_OPTIONS)
    
    def find_confirmation_button(self, parsed_content):
        """Find and click on confirmation button (yes/ok/confirm)."""
        return self.find_and_click_element(parsed_content, Config.CONFIRM_OPTIONS)
    
    def get_element_summary(self, parsed_content):
        """Get a summary of detected elements for logging."""
        if not parsed_content or not isinstance(parsed_content, list):
            return "No elements detected"
        
        element_count = len(parsed_content)
        interactive_count = sum(1 for el in parsed_content 
                              if isinstance(el, dict) and el.get('interactivity', False))
        
        return f"{element_count} total elements ({interactive_count} interactive)"