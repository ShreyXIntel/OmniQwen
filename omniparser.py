
"""
OmniParser V2 module for UI element detection and analysis - OPTIMIZED VERSION.
"""
import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import base64
import io

logger = logging.getLogger("OmniParser")

class OmniParserV2:
    """OmniParser V2 for detecting and analyzing UI elements in screenshots."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OmniParser V2 with configuration."""
        self.config = config
        self.device = config["device"]
        self.som_model = None
        self.caption_model_processor = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the OmniParser models."""
        try:
            logger.info("Initializing OmniParser V2 models...")
            
            from util.utils import (
                get_som_labeled_img,
                check_ocr_box,
                get_caption_model_processor,
                get_yolo_model,
            )
            
            self.get_som_labeled_img = get_som_labeled_img
            self.check_ocr_box = check_ocr_box
            
            # Initialize YOLO model for icon detection
            self.som_model = get_yolo_model(self.config["icon_detect_model_path"])
            self.som_model.to(self.device)
            
            # Initialize caption model (Florence2)
            self.caption_model_processor = get_caption_model_processor(
                model_name=self.config["caption_model_name"],
                model_name_or_path=self.config["caption_model_path"],
                device=self.device,
            )
            
            logger.info("OmniParser V2 models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OmniParser models: {e}")
            raise
    
    def parse_screenshot(self, image_path: str, save_output: bool = True, output_dir: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
        """Parse a screenshot to detect UI elements - OPTIMIZED VERSION."""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return [], None
            
        try:
            start_time = time.time()
            
            # Open and convert image
            image = Image.open(image_path)
            image_rgb = image.convert("RGB")
            
            # Calculate box overlay ratio based on image size
            box_overlay_ratio = max(image.size) / 3200
            draw_bbox_config = {
                "text_scale": 0.8 * box_overlay_ratio,
                "text_thickness": max(int(2 * box_overlay_ratio), 1),
                "text_padding": max(int(3 * box_overlay_ratio), 1),
                "thickness": max(int(3 * box_overlay_ratio), 1),
            }
            
            # Step 1: Run OCR to extract text elements
            ocr_bbox_rslt, is_goal_filtered = self._run_ocr(image_path)
            text, ocr_bbox = ocr_bbox_rslt
            
            # Step 2: Get labeled image with UI elements
            result = self.get_som_labeled_img(
                image_path,
                self.som_model,
                BOX_TRESHOLD=self.config["box_threshold"],
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=self.caption_model_processor,
                ocr_text=text,
                use_local_semantics=self.config["use_local_semantics"],
                iou_threshold=self.config["iou_threshold"],
                scale_img=self.config["scale_img"],
                batch_size=self.config["batch_size"],
            )
            
            # Handle the return value properly
            labeled_img_base64 = None
            label_coordinates = []
            parsed_content_list = []
            
            if isinstance(result, tuple):
                if len(result) >= 3:
                    labeled_img_base64, label_coordinates, parsed_content_list = result[:3]
                elif len(result) == 2:
                    labeled_img_base64, parsed_content_list = result
                else:
                    labeled_img_base64 = result[0] if result else None
            else:
                labeled_img_base64 = result
            
            # Save labeled image if requested
            labeled_image_path = None
            if save_output and output_dir and labeled_img_base64:
                labeled_image_path = self._save_labeled_image(
                    labeled_img_base64, image_path, output_dir
                )
            
            # Process and clean parsed content
            cleaned_content = self._process_parsed_content(parsed_content_list)
            
            parse_time = time.time() - start_time
            logger.info(f"Parsing completed in {parse_time:.2f}s with {len(cleaned_content)} elements")
            
            return cleaned_content, labeled_image_path
            
        except Exception as e:
            logger.error(f"Error in parse_screenshot: {e}")
            return [], None
    
    def _save_labeled_image(self, labeled_img_base64: str, original_path: str, output_dir: str) -> Optional[str]:
        """Save the labeled image from base64 string."""
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            analyzed_filename = f"analyzed_{base_name}_{timestamp}.png"
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            analyzed_filepath = os.path.join(output_dir, analyzed_filename)
            
            # Decode base64 image and save
            if isinstance(labeled_img_base64, str):
                # Handle base64 string
                image_data = base64.b64decode(labeled_img_base64)
                image = Image.open(io.BytesIO(image_data))
                image.save(analyzed_filepath, "PNG")
            else:
                # Handle PIL Image
                labeled_img_base64.save(analyzed_filepath, "PNG")
            
            logger.info(f"Saved analyzed image: {analyzed_filepath}")
            return analyzed_filepath
            
        except Exception as e:
            logger.error(f"Failed to save labeled image: {e}")
            return None
    
    def _run_ocr(self, image_path: str) -> Tuple[Tuple[List, List], bool]:
        """Run OCR on the image to extract text elements."""
        try:
            ocr_bbox_rslt, is_goal_filtered = self.check_ocr_box(
                image_path,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args=self.config["easyocr_args"],
                use_paddleocr=self.config["use_paddleocr"],
            )
            return ocr_bbox_rslt, is_goal_filtered
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ([], []), False
    
    def _process_parsed_content(self, parsed_content_list: List[Dict]) -> List[Dict]:
        """Process and clean the parsed content from OmniParser."""
        if not parsed_content_list or not isinstance(parsed_content_list, list):
            return []
        
        cleaned_content = []
        
        for i, element in enumerate(parsed_content_list):
            if not isinstance(element, dict):
                continue
                
            try:
                processed_element = {
                    "id": i,
                    "content": element.get("content", "").strip(),
                    "bbox": element.get("bbox", []),
                    "interactivity": element.get("interactivity", False),
                    "element_type": self._determine_element_type(element)
                }
                
                # Only include elements with valid content and bounding boxes
                if processed_element["content"] and processed_element["bbox"]:
                    cleaned_content.append(processed_element)
                    
            except Exception as e:
                logger.warning(f"Failed to process element {i}: {e}")
                continue
        
        return cleaned_content
    
    def _determine_element_type(self, element: Dict) -> str:
        """Determine the type of UI element based on its properties."""
        content = element.get("content", "").lower()
        is_interactive = element.get("interactivity", False)
        
        if is_interactive:
            if any(keyword in content for keyword in ["button", "click", "start", "begin", "run"]):
                return "button"
            elif any(keyword in content for keyword in ["menu", "option", "setting"]):
                return "menu_item"
            else:
                return "interactive"
        else:
            if any(keyword in content for keyword in ["text", "label", "title"]):
                return "text"
            elif content.isdigit() or "%" in content or "fps" in content:
                return "metric"
            else:
                return "static"
    
    def find_elements_by_keyword(self, parsed_content: List[Dict], keywords: List[str]) -> List[Dict]:
        """Find UI elements that contain specific keywords."""
        matching_elements = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for element in parsed_content:
            content = element.get("content", "").lower()
            if any(keyword in content for keyword in keywords_lower):
                matching_elements.append(element)
        
        return matching_elements
    
    def get_interactive_elements(self, parsed_content: List[Dict]) -> List[Dict]:
        """Get all interactive elements from parsed content."""
        return [elem for elem in parsed_content if elem.get("interactivity", False)]
    
    def convert_bbox_to_screen_coords(self, bbox: List[float], screen_width: int, screen_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized bounding box coordinates to screen coordinates."""
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = bbox
        
        x1 = int(x1_ratio * screen_width)
        y1 = int(y1_ratio * screen_height)
        x2 = int(x2_ratio * screen_width)
        y2 = int(y2_ratio * screen_height)
        
        return x1, y1, x2, y2
    
    def get_element_center(self, element: Dict, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """Get the center coordinates of an UI element."""
        bbox = element.get("bbox", [])
        if len(bbox) != 4:
            raise ValueError("Invalid bounding box")
        
        x1, y1, x2, y2 = self.convert_bbox_to_screen_coords(bbox, screen_width, screen_height)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        return center_x, center_y
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'som_model'):
                del self.som_model
            if hasattr(self, 'caption_model_processor'):
                del self.caption_model_processor
                
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("OmniParser resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")