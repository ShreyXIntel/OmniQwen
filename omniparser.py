"""
OmniParser V2 module for UI element detection and analysis.
"""
import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import torch
from util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)

logger = logging.getLogger("OmniParser")

class OmniParserV2:
    """OmniParser V2 for detecting and analyzing UI elements in screenshots."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OmniParser V2 with configuration.
        
        Args:
            config: Configuration dictionary for OmniParser
        """
        self.config = config
        self.device = config["device"]
        self.som_model = None
        self.caption_model_processor = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the OmniParser models."""
        try:
            logger.info("Initializing OmniParser V2 models...")
            
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
        """Parse a screenshot to detect UI elements with enhanced debugging."""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return [], None
            
        try:
            logger.info(f"Parsing screenshot: {image_path}")
            start_time = time.time()
            
            # Open and convert image
            image = Image.open(image_path)
            image_rgb = image.convert("RGB")
            logger.info(f"Image loaded: {image.size} pixels")
            
            # Enhanced box overlay ratio calculation
            box_overlay_ratio = max(image.size) / 2000  # Adjusted for better visibility
            draw_bbox_config = {
                "text_scale": max(0.6 * box_overlay_ratio, 0.5),
                "text_thickness": max(int(2 * box_overlay_ratio), 2),
                "text_padding": max(int(4 * box_overlay_ratio), 3),
                "thickness": max(int(4 * box_overlay_ratio), 3),
            }
            
            logger.info(f"Draw config: {draw_bbox_config}")
            
            # Run OCR
            ocr_bbox_rslt, is_goal_filtered = self._run_ocr(image_path)
            text, ocr_bbox = ocr_bbox_rslt
            logger.info(f"OCR found {len(text)} text elements, {len(ocr_bbox)} bboxes")
            
            # Get labeled image with UI elements
            labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
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
            
            logger.info(f"OmniParser detection: {len(parsed_content_list)} elements")
            logger.info(f"Label coordinates: {len(label_coordinates) if label_coordinates else 0}")
            logger.info(f"Labeled image type: {type(labeled_img)}")
            
            # Save labeled image if requested
            labeled_image_path = None
            if save_output:
                labeled_image_path = self._save_labeled_image(
                    labeled_img, image_rgb, image_path, output_dir
                )
            
            # Process and clean parsed content
            cleaned_content = self._process_parsed_content(parsed_content_list)
            
            parse_time = time.time() - start_time
            logger.info(f"Parsing completed in {parse_time:.2f}s with {len(cleaned_content)} elements")
            
            # Debug log element details
            for i, element in enumerate(cleaned_content[:5]):  # Log first 5 elements
                logger.info(f"Element {i}: '{element.get('content', '')}' - Interactive: {element.get('interactivity', False)}")
            
            return cleaned_content, labeled_image_path
            
        except Exception as e:
            logger.error(f"Failed to parse screenshot: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], None
    
    def _run_ocr(self, image_path: str) -> Tuple[Tuple[List, List], bool]:
        """Run OCR on the image to extract text elements.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of ((text_list, bbox_list), is_goal_filtered)
        """
        try:
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
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
    
    def _save_labeled_image(self, labeled_img: Any, fallback_image: Image.Image, 
                       original_path: str, output_dir: Optional[str]) -> Optional[str]:
        """Save the labeled image output with proper annotation handling."""
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            labeled_filename = f"labeled_{base_name}_{timestamp}.png"
            
            if output_dir:
                labeled_filepath = os.path.join(output_dir, labeled_filename)
            else:
                labeled_filepath = labeled_filename
            
            # Ensure output directory exists
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            saved = False
            
            # Method 1: Try to save the labeled image directly
            try:
                if hasattr(labeled_img, "save"):
                    labeled_img.save(labeled_filepath)
                    logger.info(f"Labeled image saved directly: {labeled_filepath}")
                    saved = True
                elif isinstance(labeled_img, str) and os.path.exists(labeled_img):
                    import shutil
                    shutil.copy2(labeled_img, labeled_filepath)
                    logger.info(f"Labeled image copied: {labeled_filepath}")
                    saved = True
                elif hasattr(labeled_img, 'shape'):  # numpy array
                    import numpy as np
                    from PIL import Image
                    
                    if isinstance(labeled_img, np.ndarray):
                        # Ensure proper format
                        if labeled_img.dtype != np.uint8:
                            labeled_img = np.clip(labeled_img * 255, 0, 255).astype(np.uint8)
                        
                        # Convert to PIL Image
                        if len(labeled_img.shape) == 3 and labeled_img.shape[2] == 3:
                            pil_img = Image.fromarray(labeled_img, 'RGB')
                        elif len(labeled_img.shape) == 3 and labeled_img.shape[2] == 4:
                            pil_img = Image.fromarray(labeled_img, 'RGBA')
                        else:
                            pil_img = Image.fromarray(labeled_img)
                        
                        pil_img.save(labeled_filepath)
                        logger.info(f"Labeled image saved from numpy array: {labeled_filepath}")
                        saved = True
            except Exception as e:
                logger.warning(f"Direct save failed: {e}")
            
            # If direct save failed, create annotated image manually
            if not saved:
                logger.warning("Creating manual annotations since OmniParser output wasn't usable")
                self._create_manual_annotations(fallback_image, labeled_filepath, original_path)
                saved = True
            
            # Verify file was saved and has reasonable size
            if saved and os.path.exists(labeled_filepath):
                file_size = os.path.getsize(labeled_filepath)
                if file_size > 1000:
                    logger.info(f"✅ Labeled image verified: {labeled_filepath} ({file_size} bytes)")
                    return labeled_filepath
            
            logger.error("Failed to save labeled image properly")
            return None
            
        except Exception as e:
            logger.error(f"Exception in _save_labeled_image: {e}")
            return None
    
    def _create_manual_annotations(self, image: Image.Image, output_path: str, original_path: str):
        """Create manual annotations when OmniParser output isn't usable."""
        try:
            from PIL import ImageDraw, ImageFont
            import numpy as np
            
            # Copy the original image
            annotated_img = image.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # Try to get font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Re-run OmniParser to get detection data
            ui_elements, _ = self.parse_screenshot(original_path, save_output=False)
            
            # Draw bounding boxes and labels
            colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
            
            for i, element in enumerate(ui_elements):
                bbox = element.get("bbox", [])
                if len(bbox) == 4:
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int(bbox[0] * image.width)
                    y1 = int(bbox[1] * image.height)
                    x2 = int(bbox[2] * image.width)
                    y2 = int(bbox[3] * image.height)
                    
                    color = colors[i % len(colors)]
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # Draw label
                    content = element.get("content", f"Element {i}")[:20]  # Limit length
                    interactive = "✓" if element.get("interactivity", False) else "✗"
                    label = f"{i}: {content} {interactive}"
                    
                    # Draw text background
                    if font:
                        bbox_text = draw.textbbox((x1, y1-20), label, font=font)
                        draw.rectangle(bbox_text, fill=color)
                        draw.text((x1, y1-20), label, fill='white', font=font)
                    else:
                        draw.text((x1, y1-10), label, fill=color)
            
            # Save annotated image
            annotated_img.save(output_path)
            logger.info(f"Manual annotations created with {len(ui_elements)} elements")
            
        except Exception as e:
            logger.error(f"Failed to create manual annotations: {e}")
            # Just save the original image as fallback
            image.save(output_path)

    def _process_parsed_content(self, parsed_content_list: List[Dict]) -> List[Dict]:
        """Process and clean the parsed content from OmniParser.
        
        Args:
            parsed_content_list: Raw parsed content from OmniParser
            
        Returns:
            Cleaned and processed content list
        """
        if not parsed_content_list or not isinstance(parsed_content_list, list):
            logger.warning("Invalid parsed content received")
            return []
        
        cleaned_content = []
        
        for i, element in enumerate(parsed_content_list):
            if not isinstance(element, dict):
                continue
                
            try:
                # Extract essential information
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
        
        logger.info(f"Processed {len(cleaned_content)} valid UI elements")
        return cleaned_content
    
    def _determine_element_type(self, element: Dict) -> str:
        """Determine the type of UI element based on its properties.
        
        Args:
            element: Element dictionary from OmniParser
            
        Returns:
            Element type string
        """
        content = element.get("content", "").lower()
        is_interactive = element.get("interactivity", False)
        
        # Determine element type based on content and interactivity
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
        """Find UI elements that contain specific keywords.
        
        Args:
            parsed_content: Parsed content from OmniParser
            keywords: List of keywords to search for
            
        Returns:
            List of matching elements
        """
        matching_elements = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for element in parsed_content:
            content = element.get("content", "").lower()
            if any(keyword in content for keyword in keywords_lower):
                matching_elements.append(element)
        
        return matching_elements
    
    def get_interactive_elements(self, parsed_content: List[Dict]) -> List[Dict]:
        """Get all interactive elements from parsed content.
        
        Args:
            parsed_content: Parsed content from OmniParser
            
        Returns:
            List of interactive elements
        """
        return [elem for elem in parsed_content if elem.get("interactivity", False)]
    
    def convert_bbox_to_screen_coords(self, bbox: List[float], screen_width: int, screen_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized bounding box coordinates to screen coordinates.
        
        Args:
            bbox: Normalized bounding box [x1_ratio, y1_ratio, x2_ratio, y2_ratio]
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            Screen coordinates (x1, y1, x2, y2)
        """
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = bbox
        
        x1 = int(x1_ratio * screen_width)
        y1 = int(y1_ratio * screen_height)
        x2 = int(x2_ratio * screen_width)
        y2 = int(y2_ratio * screen_height)
        
        return x1, y1, x2, y2
    
    def get_element_center(self, element: Dict, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """Get the center coordinates of an UI element.
        
        Args:
            element: UI element dictionary
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            Center coordinates (x, y)
        """
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