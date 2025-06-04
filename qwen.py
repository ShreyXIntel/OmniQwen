"""
Qwen 2.5 7B Instruct module for decision making in game benchmarking.
"""
import logging
import time
from typing import Dict, List, Optional, Any
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("QwenDecisionMaker")

class QwenDecisionMaker:
    """Qwen 2.5 7B Instruct model for making decisions based on UI analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Qwen decision maker.
        
        Args:
            config: Configuration dictionary for Qwen model
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # Import decision prompts from config module
        import config as config_module
        self.decision_prompts = config_module.DECISION_PROMPTS
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen 2.5 7B Instruct model."""
        try:
            logger.info("Loading Qwen 2.5 7B Instruct model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=self.config["torch_dtype"],
                device_map=self.config["device_map"]
            )
            
            logger.info("Qwen 2.5 7B Instruct model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {e}")
            raise
    
    def detect_benchmark_results(self, ui_elements: List[Dict]) -> bool:
        """Detect if benchmark results are shown using simple prompt."""
        try:
            elements_text = self._format_ui_elements_simple(ui_elements)
            prompt = self.decision_prompts["detect_results"].format(ui_elements=elements_text)
            
            response = self._generate_response(prompt)
            
            # Save debug response
            self._save_debug_response(prompt, response, "benchmark_detection")
            
            # Simple YES/NO parsing
            result = "YES" in response.upper()
            logger.info(f"Benchmark results detection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting results: {e}")
            return False
    
    def find_exit_button(self, ui_elements: List[Dict]) -> Optional[str]:
        """Find exit button using focused prompt."""
        try:
            elements_text = self._format_ui_elements_simple(ui_elements)
            prompt = self.decision_prompts["find_exit"].format(ui_elements=elements_text)
            
            response = self._generate_response(prompt)
            
            # Save debug response
            self._save_debug_response(prompt, response, "exit_detection")
            
            # Simple parsing for "CLICK: element_name"
            if "CLICK:" in response:
                element_name = response.split("CLICK:")[1].strip()
                logger.info(f"Exit button found: {element_name}")
                return element_name
            
            logger.info("No exit button found")
            return None
                
        except Exception as e:
            logger.error(f"Error finding exit button: {e}")
            return None
    
    def analyze_with_flow_context(self, ui_elements: List[Dict], step_info: Any) -> Dict[str, Any]:
        """Analyze UI elements with specific flow step context."""
        try:
            # Create focused prompt based on flow context
            prompt = f"""
                    SCREEN: {step_info.screen_context}
                    GOAL: {step_info.description}
                    LOOKING FOR: {step_info.target_button}

                    UI ELEMENTS FOUND:
                    {self._format_ui_elements_simple(ui_elements)}

                    EXPECTED BUTTONS: {', '.join(step_info.expected_buttons)}

                    Find "{step_info.target_button}" button:
                    FOUND: YES/NO
                    ELEMENT: [exact name if found]
                """
            
            logger.info(f"Qwen analyzing step: {step_info.description}")
            response = self._generate_response(prompt)
            
            # Always save the response for debugging
            self._save_debug_response(prompt, response, step_info.screen_context)
            
            # Simple parsing
            found = "YES" in response.upper()
            element_name = None
            
            if "ELEMENT:" in response:
                element_line = [line for line in response.split('\n') if 'ELEMENT:' in line]
                if element_line:
                    element_name = element_line[0].split('ELEMENT:')[1].strip()
                    if element_name.lower() in ['none', 'n/a', 'not found', '']:
                        element_name = None
            
            result = {
                "found": found,
                "element_name": element_name,
                "confidence": 0.9 if found else 0.1,
                "reasoning": f"Looking for '{step_info.target_button}' in {step_info.screen_context}",
                "raw_response": response  # Include raw response for debugging
            }
            
            logger.info(f"Qwen result: Found={found}, Element='{element_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error in flow context analysis: {e}")
            return {"found": False, "element_name": None, "confidence": 0.0, "reasoning": "Analysis failed"}
    
    def _format_ui_elements_simple(self, ui_elements: List[Dict]) -> str:
        """Format UI elements in a simple, focused way.
        
        Args:
            ui_elements: List of UI elements
            
        Returns:
            Simple formatted string
        """
        if not ui_elements:
            return "No UI elements detected"
        
        elements = []
        for element in ui_elements[:10]:  # Limit to first 10 elements
            content = element.get("content", "").strip()
            if content and len(content) > 0:
                interactive = "✓" if element.get("interactivity", False) else "✗"
                elements.append(f"- {content} ({interactive})")
        
        return "\n".join(elements) if elements else "No readable elements found"
    
    def _format_ui_elements(self, ui_elements: List[Dict]) -> str:
        """Format UI elements for inclusion in prompts.
        
        Args:
            ui_elements: List of UI elements
            
        Returns:
            Formatted string representation of UI elements
        """
        if not ui_elements:
            return "No UI elements detected."
        
        formatted_elements = []
        for i, element in enumerate(ui_elements):
            content = element.get("content", "Unknown")
            interactive = element.get("interactivity", False)
            element_type = element.get("element_type", "unknown")
            bbox = element.get("bbox", [])
            
            # Format bounding box
            bbox_str = f"[{', '.join(map(str, bbox))}]" if bbox else "[no coordinates]"
            
            # Create element description
            interaction_status = "Interactive" if interactive else "Static"
            element_desc = f"{i+1}. \"{content}\" - {element_type} ({interaction_status}) {bbox_str}"
            formatted_elements.append(element_desc)
        
        return "\n".join(formatted_elements)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from Qwen model.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated response text
        """
        try:
            # Prepare messages for chat template
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in game UI analysis and decision making."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    repetition_penalty=self.config["repetition_penalty"],
                    do_sample=self.config["do_sample"],
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "ERROR: Failed to generate response"
    
    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse decision response from Qwen model.
        
        Args:
            response: Raw response from model
            
        Returns:
            Parsed decision dictionary
        """
        try:
            decision = {
                "action": None,
                "target": None,
                "confidence": 0.0,
                "reasoning": ""
            }
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("ACTION:"):
                    decision["action"] = line.split(":", 1)[1].strip()
                elif line.startswith("TARGET:"):
                    target = line.split(":", 1)[1].strip()
                    decision["target"] = target if target != "N/A" else None
                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf_str = line.split(":", 1)[1].strip()
                        decision["confidence"] = float(conf_str)
                    except ValueError:
                        decision["confidence"] = 0.5
                elif line.startswith("REASONING:"):
                    decision["reasoning"] = line.split(":", 1)[1].strip()
            
            # Validate action
            if decision["action"] not in ["CLICK", "WAIT", "BACK", "EXIT"]:
                decision = self._create_default_decision()
            
            return decision
            
        except Exception as e:
            logger.error(f"Error parsing decision response: {e}")
            return self._create_default_decision()
    
    def _parse_detection_response(self, response: str) -> Dict[str, Any]:
        """Parse benchmark detection response.
        
        Args:
            response: Raw response from model
            
        Returns:
            Parsed detection dictionary
        """
        try:
            detection = {
                "found": False,
                "element_name": None,
                "confidence": 0.0,
                "reasoning": ""
            }
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("BENCHMARK_FOUND:"):
                    found_str = line.split(":", 1)[1].strip().upper()
                    detection["found"] = found_str == "YES"
                elif line.startswith("ELEMENT_NAME:"):
                    element = line.split(":", 1)[1].strip()
                    detection["element_name"] = element if element != "N/A" else None
                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf_str = line.split(":", 1)[1].strip()
                        detection["confidence"] = float(conf_str)
                    except ValueError:
                        detection["confidence"] = 0.5
                elif line.startswith("REASONING:"):
                    detection["reasoning"] = line.split(":", 1)[1].strip()
            
            return detection
            
        except Exception as e:
            logger.error(f"Error parsing detection response: {e}")
            return {"found": False, "element_name": None, "confidence": 0.0, "reasoning": "Parse error"}
    
    def _parse_result_detection_response(self, response: str) -> Dict[str, Any]:
        """Parse benchmark result detection response.
        
        Args:
            response: Raw response from model
            
        Returns:
            Parsed result detection dictionary
        """
        try:
            detection = {
                "found": False,
                "confidence": 0.0,
                "metrics": [],
                "reasoning": ""
            }
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("RESULTS_FOUND:"):
                    found_str = line.split(":", 1)[1].strip().upper()
                    detection["found"] = found_str == "YES"
                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf_str = line.split(":", 1)[1].strip()
                        detection["confidence"] = float(conf_str)
                    except ValueError:
                        detection["confidence"] = 0.5
                elif line.startswith("METRICS_DETECTED:"):
                    metrics_str = line.split(":", 1)[1].strip()
                    if metrics_str and metrics_str != "N/A":
                        detection["metrics"] = [m.strip() for m in metrics_str.split(',')]
                elif line.startswith("REASONING:"):
                    detection["reasoning"] = line.split(":", 1)[1].strip()
            
            return detection
            
        except Exception as e:
            logger.error(f"Error parsing result detection response: {e}")
            return {"found": False, "confidence": 0.0, "metrics": [], "reasoning": "Parse error"}
    
    def _create_default_decision(self) -> Dict[str, Any]:
        """Create a default/fallback decision.
        
        Returns:
            Default decision dictionary
        """
        return {
            "action": "WAIT",
            "target": None,
            "confidence": 0.3,
            "reasoning": "Default action due to analysis failure"
        }
    
    def save_response(self, prompt: str, response: str, output_dir: str, filename_prefix: str = "qwen_response"):
        """Save Qwen response to file for debugging.
        
        Args:
            prompt: Original prompt
            response: Model response
            output_dir: Directory to save the response
            filename_prefix: Prefix for the filename
        """
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n=== RESPONSE ===\n")
                f.write(response)
            
            logger.info(f"Qwen response saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save Qwen response: {e}")
    
    def _save_debug_response(self, prompt: str, response: str, context: str = "unknown"):
        """Save debug response automatically."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            
            # Create a debug directory if it doesn't exist
            debug_dir = "debug_qwen_responses"
            os.makedirs(debug_dir, exist_ok=True)
            
            filename = f"qwen_{context}_{timestamp}.txt"
            filepath = os.path.join(debug_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write(f"TIMESTAMP: {timestamp}\n")
                f.write(f"CONTEXT: {context}\n")
                f.write("="*60 + "\n\n")
                f.write("PROMPT:\n")
                f.write("-" * 40 + "\n")
                f.write(prompt)
                f.write("\n\n")
                f.write("RESPONSE:\n")
                f.write("-" * 40 + "\n")
                f.write(response)
                f.write("\n\n")
                f.write("="*60 + "\n")
            
            logger.info(f"Debug response saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save debug response: {e}")

    def cleanup(self):
        """Clean up model resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Qwen model resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during Qwen cleanup: {e}")