"""
Enhanced Qwen 2.5 7B Instruct module with flow-aware decision making.
Now understands game-specific flow progression and provides better context-aware decisions.
"""
import logging
import time
import os
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("QwenDecisionMaker")

class QwenDecisionMaker:
    """Enhanced Qwen 2.5 7B Instruct model with flow-aware decision making."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced Qwen decision maker.
        
        Args:
            config: Configuration dictionary for Qwen model
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.prompt_counter = 0
        self.response_counter = 0
        
        # Enhanced prompt templates for flow-aware decision making
        self.flow_aware_prompts = {
            "flow_navigation": """
FLOW CONTEXT:
- Current State: {current_state}
- Expected Next States: {expected_next_states}
- Goal State: {goal_state}
- Flow Progress: {flow_progress}
- Game: {game_name}

CURRENT SITUATION:
{situation_description}

UI ELEMENTS:
{ui_elements}

GAME-SPECIFIC KEYWORDS:
- Benchmark Triggers: {benchmark_triggers}
- Navigation Helpers: {navigation_helpers}
- Confirmation Words: {confirmation_words}

Based on the flow context and current state, determine the best action to progress toward the goal.
Consider:
1. Are we on the expected flow path?
2. Which UI elements match our expected next states?
3. What action will move us closer to the goal?

Respond with:
ACTION: CLICK/WAIT/BACK
TARGET: [element name if clicking]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation considering flow context]
""",
            
            "state_detection": """
GAME: {game_name}
FLOW SEQUENCE: {flow_sequence}

CURRENT UI ELEMENTS:
{ui_elements}

STATE KEYWORDS FOR THIS GAME:
{state_keywords}

Based on the UI elements and game-specific keywords, which state are we most likely in?
Consider:
1. Which keywords appear in the UI elements?
2. Which state best matches the current screen?
3. Where are we in the expected flow sequence?

Respond with:
STATE: [state_name]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation of why this state was chosen]
""",
            
            "benchmark_detection": """
GAME: {game_name}
GOAL: Find and start the benchmark

UI ELEMENTS:
{ui_elements}

BENCHMARK KEYWORDS FOR THIS GAME: {benchmark_keywords}
NAVIGATION KEYWORDS: {navigation_keywords}

Look for benchmark-related elements or navigation options to reach the benchmark.
Consider game-specific terminology and variations.

Respond with:
FOUND: YES/NO
ELEMENT: [element name if found]
ACTION: CLICK/NAVIGATE/WAIT
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]
""",
            
            "result_detection": """
GAME: {game_name}
CHECKING FOR: Benchmark results or completion

UI ELEMENTS:
{ui_elements}

RESULT INDICATORS FOR THIS GAME: {result_indicators}
COMPLETION KEYWORDS: {completion_keywords}

Look for signs that the benchmark has completed and results are displayed.
Consider game-specific result formats and terminology.

Respond with:
RESULTS_FOUND: YES/NO
CONFIDENCE: [0.0-1.0]
METRICS_DETECTED: [list any FPS/performance numbers found]
REASONING: [explanation]
"""
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen 2.5 7B Instruct model."""
        try:
            logger.info("Loading Enhanced Qwen 2.5 7B Instruct model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=self.config["torch_dtype"],
                device_map=self.config["device_map"]
            )
            
            logger.info("Enhanced Qwen 2.5 7B Instruct model loaded successfully")
            logger.info("Features: Flow-aware prompts, game-specific context, enhanced reasoning")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced Qwen model: {e}")
            raise
    
    def make_flow_aware_decision(self, flow_context: Dict, ui_elements: List[Dict], 
                                game_config: Dict) -> Dict[str, Any]:
        """Make a flow-aware decision based on current context.
        
        Args:
            flow_context: Current flow context information
            ui_elements: UI elements from OmniParser
            game_config: Game-specific configuration
            
        Returns:
            Decision dictionary with action, target, confidence, and reasoning
        """
        try:
            # Extract game-specific keywords
            priority_keywords = game_config.get('priority_keywords', {})
            benchmark_triggers = priority_keywords.get('benchmark_triggers', ['benchmark'])
            navigation_helpers = priority_keywords.get('navigation_helpers', ['options', 'settings'])
            confirmation_words = priority_keywords.get('confirmation_words', ['yes', 'confirm'])
            
            # Format UI elements
            ui_elements_text = self._format_ui_elements_enhanced(ui_elements)
            
            # Create flow-aware prompt
            prompt = self.flow_aware_prompts["flow_navigation"].format(
                current_state=flow_context.get('current_state', 'unknown'),
                expected_next_states=', '.join(flow_context.get('expected_next_states', [])),
                goal_state=flow_context.get('goal_state', 'benchmark_results'),
                flow_progress=flow_context.get('flow_progress', '0/0'),
                game_name=flow_context.get('game_name', 'unknown'),
                situation_description=flow_context.get('situation', 'Navigating through game menus'),
                ui_elements=ui_elements_text,
                benchmark_triggers=', '.join(benchmark_triggers),
                navigation_helpers=', '.join(navigation_helpers),
                confirmation_words=', '.join(confirmation_words)
            )
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Parse enhanced response
            decision = self._parse_enhanced_decision_response(response)
            
            # Add flow context to decision
            decision['flow_context'] = flow_context
            decision['prompt_type'] = 'flow_aware_navigation'
            
            logger.info(f"Flow-aware decision: {decision['action']} (confidence: {decision.get('confidence', 0):.2f})")
            logger.debug(f"Flow reasoning: {decision.get('reasoning', 'No reasoning provided')}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in flow-aware decision making: {e}")
            return self._create_fallback_decision("Flow-aware decision failed")
    
    def detect_state_with_flow_context(self, ui_elements: List[Dict], flow_sequence: List[str],
                                     state_keywords: Dict, game_name: str) -> Dict[str, Any]:
        """Detect current state using flow context and game-specific keywords.
        
        Args:
            ui_elements: UI elements from OmniParser
            flow_sequence: Expected flow sequence for the game
            state_keywords: Game-specific state detection keywords
            game_name: Name of the current game
            
        Returns:
            State detection result with confidence and reasoning
        """
        try:
            # Format inputs
            ui_elements_text = self._format_ui_elements_enhanced(ui_elements)
            flow_sequence_text = ' -> '.join(flow_sequence)
            
            # Format state keywords
            state_keywords_text = ""
            for state_name, keywords in state_keywords.items():
                state_keywords_text += f"{state_name}: {', '.join(keywords)}\n"
            
            # Create state detection prompt
            prompt = self.flow_aware_prompts["state_detection"].format(
                game_name=game_name,
                flow_sequence=flow_sequence_text,
                ui_elements=ui_elements_text,
                state_keywords=state_keywords_text
            )
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Parse state detection response
            result = self._parse_state_detection_response(response)
            
            logger.info(f"State detection: {result.get('state', 'unknown')} (confidence: {result.get('confidence', 0):.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in state detection: {e}")
            return {"state": "unknown", "confidence": 0.0, "reasoning": f"Detection failed: {str(e)}"}
    
    def detect_benchmark_with_game_context(self, ui_elements: List[Dict], game_config: Dict) -> Dict[str, Any]:
        """Detect benchmark options using game-specific context.
        
        Args:
            ui_elements: UI elements from OmniParser
            game_config: Game-specific configuration
            
        Returns:
            Benchmark detection result
        """
        try:
            game_name = game_config.get('name', 'unknown')
            priority_keywords = game_config.get('priority_keywords', {})
            
            benchmark_keywords = priority_keywords.get('benchmark_triggers', ['benchmark', 'test', 'performance'])
            navigation_keywords = priority_keywords.get('navigation_helpers', ['options', 'settings', 'graphics'])
            
            # Format UI elements
            ui_elements_text = self._format_ui_elements_enhanced(ui_elements)
            
            # Create benchmark detection prompt
            prompt = self.flow_aware_prompts["benchmark_detection"].format(
                game_name=game_name,
                ui_elements=ui_elements_text,
                benchmark_keywords=', '.join(benchmark_keywords),
                navigation_keywords=', '.join(navigation_keywords)
            )
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Parse benchmark detection response
            result = self._parse_benchmark_detection_response(response)
            
            logger.info(f"Benchmark detection: found={result.get('found', False)} (confidence: {result.get('confidence', 0):.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in benchmark detection: {e}")
            return {"found": False, "confidence": 0.0, "reasoning": f"Detection failed: {str(e)}"}
    
    def detect_benchmark_results(self, ui_elements: List[Dict], game_config: Optional[Dict] = None) -> bool:
        """Enhanced benchmark result detection with game-specific context."""
        try:
            if game_config:
                return self._detect_results_with_game_context(ui_elements, game_config)
            else:
                return self._detect_results_generic(ui_elements)
                
        except Exception as e:
            logger.error(f"Error detecting benchmark results: {e}")
            return False
    
    def _detect_results_with_game_context(self, ui_elements: List[Dict], game_config: Dict) -> bool:
        """Detect results using game-specific context."""
        try:
            game_name = game_config.get('name', 'unknown')
            
            # Get game-specific result indicators
            state_keywords = game_config.get('state_keywords', {})
            result_indicators = state_keywords.get('BENCHMARK_RESULTS', [
                'results', 'score', 'fps', 'average', 'minimum', 'maximum', 'finished'
            ])
            
            completion_keywords = [
                'benchmark complete', 'test finished', 'results', 'summary', 'finished', 'completed'
            ]
            
            # Format UI elements
            ui_elements_text = self._format_ui_elements_enhanced(ui_elements)
            
            # Create result detection prompt
            prompt = self.flow_aware_prompts["result_detection"].format(
                game_name=game_name,
                ui_elements=ui_elements_text,
                result_indicators=', '.join(result_indicators),
                completion_keywords=', '.join(completion_keywords)
            )
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Parse result detection response
            result = self._parse_result_detection_response(response)
            
            found = result.get('results_found', False)
            confidence = result.get('confidence', 0.0)
            
            logger.info(f"Game-specific result detection: found={found} (confidence: {confidence:.2f})")
            
            return found
            
        except Exception as e:
            logger.error(f"Error in game-specific result detection: {e}")
            return False
    
    def _detect_results_generic(self, ui_elements: List[Dict]) -> bool:
        """Fallback generic result detection."""
        try:
            elements_text = self._format_ui_elements_enhanced(ui_elements)
            
            prompt = f"""
UI Elements (may contain OCR errors or partial text):
{elements_text}

Look for benchmark RESULTS or COMPLETION indicators. OCR may have corrupted text, so consider:
- "FPS", "fps" might appear as "PS", "fp", "FS"  
- "RESULTS" might appear as "RSLT", "RESULT", "RES"
- "AVERAGE" might appear as "AVG", "AVERG", "AVER"
- "FINISHED" might appear as "FIN", "FINSHD", "DONE"
- "COMPLETE" might appear as "CMPL", "COMP", "COMPLET"
- Numbers with % or numerical scores
- "BENCHMARK" might appear as "BNCH", "BNCHMRK", "BENCH"

Are benchmark results visible? Answer: YES or NO
"""
            
            response = self._generate_response(prompt)
            
            # Simple YES/NO parsing
            return "YES" in response.upper()
            
        except Exception as e:
            logger.error(f"Error in generic result detection: {e}")
            return False
    
    def _format_ui_elements_enhanced(self, ui_elements: List[Dict]) -> str:
        """Enhanced UI element formatting with more context."""
        if not ui_elements:
            return "No UI elements detected"
        
        elements = []
        interactive_count = 0
        text_count = 0
        
        for i, element in enumerate(ui_elements[:12]):  # Limit to first 12 elements
            content = element.get("content", "").strip()
            if content and len(content) > 0:
                interactive = element.get("interactivity", False)
                element_type = element.get("element_type", "unknown")
                
                if interactive:
                    interactive_count += 1
                    elements.append(f"- '{content}' [INTERACTIVE-{element_type.upper()}]")
                else:
                    text_count += 1
                    elements.append(f"- '{content}' [TEXT-{element_type.upper()}]")
        
        if not elements:
            return "No readable elements found"
        
        header = f"Found {len(elements)} elements ({interactive_count} interactive, {text_count} text):\n"
        note = "NOTE: Text may contain OCR errors or be partially corrupted\n\n"
        return header + note + "\n".join(elements)
    
    def _parse_enhanced_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse enhanced decision response with better error handling."""
        try:
            decision = {
                "action": "WAIT",
                "target": None,
                "confidence": 0.5,
                "reasoning": "Default decision"
            }
            
            lines = response.split('\n')
            for line in lines:
                line_clean = line.strip()
                line_upper = line_clean.upper()
                
                if line_upper.startswith("ACTION:"):
                    action_part = line_upper.split(":", 1)[1].strip()
                    # More flexible action parsing
                    if "CLICK" in action_part:
                        decision["action"] = "CLICK"
                    elif "WAIT" in action_part:
                        decision["action"] = "WAIT"
                    elif "BACK" in action_part:
                        decision["action"] = "BACK"
                    elif "NAVIGATE" in action_part:
                        decision["action"] = "CLICK"  # Treat navigate as click
                        
                elif line_upper.startswith("TARGET:"):
                    target = line_clean.split(":", 1)[1].strip()
                    if target and target.upper() not in ["N/A", "NONE", "NULL", ""]:
                        decision["target"] = target
                        
                elif line_upper.startswith("CONFIDENCE:"):
                    try:
                        conf_part = line_clean.split(":", 1)[1].strip()
                        # Extract numerical value with regex
                        import re
                        numbers = re.findall(r'[\d.]+', conf_part)
                        if numbers:
                            conf_val = float(numbers[0])
                            if conf_val > 1.0:
                                conf_val = conf_val / 100.0
                            decision["confidence"] = min(max(conf_val, 0.0), 1.0)
                    except (ValueError, IndexError):
                        pass
                        
                elif line_upper.startswith("REASONING:"):
                    reasoning = line_clean.split(":", 1)[1].strip()
                    if reasoning:
                        decision["reasoning"] = reasoning
            
            return decision
            
        except Exception as e:
            logger.error(f"Error parsing enhanced decision response: {e}")
            return self._create_fallback_decision("Response parsing failed")
    
    def _parse_state_detection_response(self, response: str) -> Dict[str, Any]:
        """Parse state detection response."""
        try:
            result = {
                "state": "unknown",
                "confidence": 0.0,
                "reasoning": "No analysis"
            }
            
            lines = response.split('\n')
            for line in lines:
                line_clean = line.strip()
                line_upper = line_clean.upper()
                
                if line_upper.startswith("STATE:"):
                    state = line_clean.split(":", 1)[1].strip()
                    if state:
                        result["state"] = state.lower()
                        
                elif line_upper.startswith("CONFIDENCE:"):
                    try:
                        conf_part = line_clean.split(":", 1)[1].strip()
                        import re
                        numbers = re.findall(r'[\d.]+', conf_part)
                        if numbers:
                            conf_val = float(numbers[0])
                            if conf_val > 1.0:
                                conf_val = conf_val / 100.0
                            result["confidence"] = min(max(conf_val, 0.0), 1.0)
                    except (ValueError, IndexError):
                        pass
                        
                elif line_upper.startswith("REASONING:"):
                    reasoning = line_clean.split(":", 1)[1].strip()
                    if reasoning:
                        result["reasoning"] = reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing state detection response: {e}")
            return {"state": "unknown", "confidence": 0.0, "reasoning": f"Parse error: {str(e)}"}
    
    def _parse_benchmark_detection_response(self, response: str) -> Dict[str, Any]:
        """Parse benchmark detection response."""
        try:
            result = {
                "found": False,
                "element": None,
                "action": "WAIT",
                "confidence": 0.0,
                "reasoning": "No analysis"
            }
            
            lines = response.split('\n')
            for line in lines:
                line_clean = line.strip()
                line_upper = line_clean.upper()
                
                if line_upper.startswith("FOUND:"):
                    found_str = line_upper.split(":", 1)[1].strip()
                    result["found"] = "YES" in found_str
                    
                elif line_upper.startswith("ELEMENT:"):
                    element = line_clean.split(":", 1)[1].strip()
                    if element and element.upper() not in ["N/A", "NONE", "NULL"]:
                        result["element"] = element
                        
                elif line_upper.startswith("ACTION:"):
                    action_part = line_upper.split(":", 1)[1].strip()
                    if "CLICK" in action_part:
                        result["action"] = "CLICK"
                    elif "NAVIGATE" in action_part:
                        result["action"] = "CLICK"
                    elif "WAIT" in action_part:
                        result["action"] = "WAIT"
                        
                elif line_upper.startswith("CONFIDENCE:"):
                    try:
                        conf_part = line_clean.split(":", 1)[1].strip()
                        import re
                        numbers = re.findall(r'[\d.]+', conf_part)
                        if numbers:
                            conf_val = float(numbers[0])
                            if conf_val > 1.0:
                                conf_val = conf_val / 100.0
                            result["confidence"] = min(max(conf_val, 0.0), 1.0)
                    except (ValueError, IndexError):
                        pass
                        
                elif line_upper.startswith("REASONING:"):
                    reasoning = line_clean.split(":", 1)[1].strip()
                    if reasoning:
                        result["reasoning"] = reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing benchmark detection response: {e}")
            return {"found": False, "confidence": 0.0, "reasoning": f"Parse error: {str(e)}"}
    
    def _parse_result_detection_response(self, response: str) -> Dict[str, Any]:
        """Parse result detection response."""
        try:
            result = {
                "results_found": False,
                "confidence": 0.0,
                "metrics": [],
                "reasoning": "No analysis"
            }
            
            lines = response.split('\n')
            for line in lines:
                line_clean = line.strip()
                line_upper = line_clean.upper()
                
                if line_upper.startswith("RESULTS_FOUND:"):
                    found_str = line_upper.split(":", 1)[1].strip()
                    result["results_found"] = "YES" in found_str
                    
                elif line_upper.startswith("CONFIDENCE:"):
                    try:
                        conf_part = line_clean.split(":", 1)[1].strip()
                        import re
                        numbers = re.findall(r'[\d.]+', conf_part)
                        if numbers:
                            conf_val = float(numbers[0])
                            if conf_val > 1.0:
                                conf_val = conf_val / 100.0
                            result["confidence"] = min(max(conf_val, 0.0), 1.0)
                    except (ValueError, IndexError):
                        pass
                        
                elif line_upper.startswith("METRICS_DETECTED:"):
                    metrics_str = line_clean.split(":", 1)[1].strip()
                    if metrics_str and metrics_str.upper() not in ["N/A", "NONE", "NULL"]:
                        # Simple parsing of metrics
                        metrics = [m.strip() for m in metrics_str.split(',')]
                        result["metrics"] = metrics
                        
                elif line_upper.startswith("REASONING:"):
                    reasoning = line_clean.split(":", 1)[1].strip()
                    if reasoning:
                        result["reasoning"] = reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing result detection response: {e}")
            return {"results_found": False, "confidence": 0.0, "reasoning": f"Parse error: {str(e)}"}
    
    def _create_fallback_decision(self, reason: str) -> Dict[str, Any]:
        """Create a fallback decision when analysis fails."""
        return {
            "action": "WAIT",
            "target": None,
            "confidence": 0.3,
            "reasoning": f"Fallback decision: {reason}"
        }
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from Qwen model with enhanced error handling."""
        try:
            # Prepare messages for chat template
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in game UI analysis and flow-aware decision making. You understand game-specific terminology and can navigate complex UI flows. You can interpret partially corrupted OCR text and understand flow progression context."},
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
    
    def save_enhanced_prompt_and_response(self, prompt: str, response: str, prompt_dir: str, 
                                        response_dir: str, context: str = "enhanced_analysis"):
        """Save enhanced Qwen prompt and response with additional metadata."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.prompt_counter += 1
            self.response_counter += 1
            
            # Ensure directories exist
            os.makedirs(prompt_dir, exist_ok=True)
            os.makedirs(response_dir, exist_ok=True)
            
            # Save enhanced prompt
            prompt_filename = f"{context}_prompt_{self.prompt_counter:04d}_{timestamp}.txt"
            prompt_filepath = os.path.join(prompt_dir, prompt_filename)
            
            with open(prompt_filepath, 'w', encoding='utf-8') as f:
                f.write("=== ENHANCED QWEN PROMPT ===\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Context: {context}\n")
                f.write(f"Prompt ID: {self.prompt_counter}\n")
                f.write(f"Version: Enhanced Flow-Aware\n")
                f.write(f"Features: Game-specific context, Flow progression awareness\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write(f"\n=== PROMPT CONTENT ===\n")
                f.write(prompt)
                f.write(f"\n\n=== END PROMPT ===\n")
            
            # Save enhanced response
            response_filename = f"{context}_response_{self.response_counter:04d}_{timestamp}.txt"
            response_filepath = os.path.join(response_dir, response_filename)
            
            with open(response_filepath, 'w', encoding='utf-8') as f:
                f.write("=== ENHANCED QWEN RESPONSE ===\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Context: {context}\n")
                f.write(f"Response ID: {self.response_counter}\n")
                f.write(f"Corresponding Prompt: {prompt_filename}\n")
                f.write(f"Version: Enhanced Flow-Aware\n")
                f.write(f"Response Length: {len(response)} characters\n")
                f.write(f"\n=== RESPONSE CONTENT ===\n")
                f.write(response)
                f.write(f"\n\n=== END RESPONSE ===\n")
            
            logger.info(f"Enhanced Qwen prompt saved: {prompt_filepath}")
            logger.info(f"Enhanced Qwen response saved: {response_filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced Qwen prompt/response: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.config["model_name"],
            "device": self.config["device"],
            "version": "Enhanced Flow-Aware",
            "features": [
                "Game-specific context understanding",
                "Flow progression awareness", 
                "Enhanced prompt templates",
                "Multi-method result detection",
                "Improved error handling"
            ],
            "prompt_count": self.prompt_counter,
            "response_count": self.response_counter
        }
    
    def cleanup(self):
        """Clean up model resources with enhanced logging."""
        try:
            logger.info(f"Cleaning up Enhanced Qwen model (processed {self.response_counter} responses)")
            
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Enhanced Qwen model resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during enhanced Qwen cleanup: {e}")