"""
Computer Vision and Image Detection Module
Handles OmniParser setup, image capture, and UI element detection with debug annotations
"""

import os
import time
import datetime
import logging
from PIL import Image, ImageGrab
from config import Config
from debug_annotator import DebugAnnotator
from util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)


class VisionDetector:
    def __init__(self, model_path=None):
        self.logger = logging.getLogger(Config.LOGGER_NAME)
        
        # Set model path
        self.model_path = model_path or Config.MODEL_PATH
        
        # Set up device
        self.device = Config.DEVICE
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize OmniParser components
        self.som_model = None
        self.caption_model_processor = None
        
        # Initialize debug annotator
        self.debug_annotator = DebugAnnotator() if Config.ENABLE_DEBUG_ANNOTATIONS else None
        
        # Tracking variables
        self.last_snapshot_time = 0
        self.snap_counter = 0
        
        # Setup OmniParser
        self.setup_omniparser()
    
    def setup_omniparser(self):
        """Set up OmniParserV2 for image parsing."""
        try:
            self.logger.info("Setting up OmniParserV2...")
            
            # Load YOLO model for UI element detection
            self.som_model = get_yolo_model(self.model_path)
            self.som_model.to(self.device)
            
            # Load caption model for element description
            self.caption_model_processor = get_caption_model_processor(
                model_name=Config.CAPTION_MODEL_NAME,
                model_name_or_path=Config.CAPTION_MODEL_PATH,
                device=self.device,
            )
            
            self.logger.info("OmniParserV2 setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to set up OmniParserV2: {str(e)}")
            raise
    
    def take_snapshot(self):
        """Capture a screenshot and save it."""
        try:
            # Capture the screen
            screenshot = ImageGrab.grab()
            
            # Generate filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.snap_counter += 1
            filename = f"snap_{self.snap_counter}_{timestamp}.png"
            filepath = os.path.join(Config.SNAPSHOT_DIR, filename)
            
            # Save the screenshot
            screenshot.save(filepath)
            self.logger.info(f"Snapshot saved to {filepath}")
            
            # Update last snapshot time
            self.last_snapshot_time = time.time()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to take snapshot: {str(e)}")
            return None
    
    def create_debug_annotation(self, image_path, parsed_content):
        """Manually create debug annotations for an image and parsed content."""
        if not self.debug_annotator:
            self.logger.warning("Debug annotator not initialized")
            return None
        
        try:
            # Create debug annotation
            annotated_path = self.debug_annotator.annotate_image(image_path, parsed_content)
            
            # Create elements list
            if Config.SAVE_ELEMENTS_LIST:
                self.debug_annotator.annotate_elements_list(parsed_content)
            
            # Create comparison if enabled
            if Config.CREATE_COMPARISON_IMAGES and annotated_path:
                self.debug_annotator.create_comparison_image(image_path, annotated_path)
            
            return annotated_path
            
        except Exception as e:
            self.logger.error(f"Failed to create debug annotation: {str(e)}")
            return None
    
    def should_take_snapshot(self):
        """Check if it's time to take a new snapshot."""
        return (
            time.time() - self.last_snapshot_time > Config.SNAPSHOT_INTERVAL
            or self.last_snapshot_time == 0
        )
    
    def parse_image(self, image_path):
        """Parse an image using OmniParserV2 and return labels."""
        if not image_path:
            self.logger.error("Cannot parse image: image_path is None")
            return [], None
        
        try:
            self.logger.info(f"Parsing image: {image_path}")
            
            # Open the image to get its size for overlay ratio calculation
            image = Image.open(image_path)
            image_rgb = image.convert("RGB")
            
            # Calculate box overlay ratio based on image size
            box_overlay_ratio = max(image.size) / Config.BOX_OVERLAY_RATIO_DIVISOR
            draw_bbox_config = Config.get_bbox_draw_config(box_overlay_ratio)
            
            # Run OCR
            try:
                ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                    image_path,
                    display_img=False,
                    output_bb_format="xyxy",
                    goal_filtering=None,
                    easyocr_args={
                        "paragraph": Config.OCR_PARAGRAPH_MODE,
                        "text_threshold": Config.OCR_TEXT_THRESHOLD
                    },
                    use_paddleocr=Config.OCR_USE_PADDLE,
                )
                text, ocr_bbox = ocr_bbox_rslt
            except Exception as e:
                self.logger.error(f"OCR failed: {str(e)}")
                text, ocr_bbox = [], []
            
            # Get labeled image
            try:
                labeled_img, label_coordinates, parsed_content_list = (
                    get_som_labeled_img(
                        image_path,
                        self.som_model,
                        BOX_TRESHOLD=Config.BOX_THRESHOLD,
                        output_coord_in_ratio=True,
                        ocr_bbox=ocr_bbox,
                        draw_bbox_config=draw_bbox_config,
                        caption_model_processor=self.caption_model_processor,
                        ocr_text=text,
                        use_local_semantics=Config.USE_LOCAL_SEMANTICS,
                        iou_threshold=Config.IOU_THRESHOLD,
                        scale_img=Config.SCALE_IMG,
                        batch_size=Config.BATCH_SIZE,
                    )
                )
            except Exception as e:
                self.logger.error(f"Failed to get labeled image: {str(e)}")
                return [], None
            
            # Save the labeled image
            labeled_filepath = self._save_labeled_image(labeled_img, image_rgb)
            
            # Validate parsed content
            if not parsed_content_list or not isinstance(parsed_content_list, list):
                self.logger.warning(
                    f"Invalid parsed content, using empty list: {type(parsed_content_list)}"
                )
                parsed_content_list = []
            
            # Log the parsed content
            self.logger.info(f"Image parsed with {len(parsed_content_list)} elements")
            
            # Create debug annotations if enabled
            debug_annotated_path = None
            if self.debug_annotator and Config.ENABLE_DEBUG_ANNOTATIONS:
                debug_annotated_path = self.debug_annotator.annotate_image(
                    image_path, parsed_content_list
                )
                
                # Save elements list if enabled
                if Config.SAVE_ELEMENTS_LIST:
                    self.debug_annotator.annotate_elements_list(parsed_content_list)
                
                # Create comparison image if enabled
                if Config.CREATE_COMPARISON_IMAGES and debug_annotated_path:
                    self.debug_annotator.create_comparison_image(
                        image_path, debug_annotated_path
                    )
            
            return parsed_content_list, labeled_filepath
            
        except Exception as e:
            self.logger.error(f"Failed to parse image: {str(e)}")
            return [], None
    
    def _save_labeled_image(self, labeled_img, image_rgb):
        """Save the labeled image to the appropriate directory."""
        # Generate timestamp and filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        labeled_filename = f"labeled_{self.snap_counter}_{timestamp}.png"
        labeled_filepath = os.path.join(Config.PARSED_IMAGES_DIR, labeled_filename)
        
        try:
            # Check if labeled_img is a PIL Image object or string path
            if hasattr(labeled_img, "save"):
                # It's a PIL Image object
                labeled_img.save(labeled_filepath)
                self.logger.info(f"Labeled image saved to {labeled_filepath}")
            elif isinstance(labeled_img, str) and os.path.isfile(labeled_img):
                # It's a file path string
                import shutil
                shutil.copy(labeled_img, labeled_filepath)
                self.logger.info(f"Labeled image copied from {labeled_img} to {labeled_filepath}")
            else:
                self.logger.warning(
                    f"Labeled image not saved - not a valid image object: {type(labeled_img)}"
                )
                # Store original image as fallback
                image_rgb.save(labeled_filepath)
                self.logger.info(f"Original image saved as fallback to {labeled_filepath}")
        
        except Exception as e:
            self.logger.error(f"Failed to save labeled image: {str(e)}")
            # Use original image as fallback
            image_rgb.save(labeled_filepath)
            
        return labeled_filepath
    
    def take_benchmark_result_snapshot(self):
        """Take a special snapshot for benchmark results."""
        try:
            result_screenshot = ImageGrab.grab()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"benchmark_result_{timestamp}.png"
            result_filepath = os.path.join(Config.BENCHMARK_RESULTS_DIR, result_filename)
            result_screenshot.save(result_filepath)
            self.logger.info(f"Benchmark results saved to {result_filepath}")
            return result_filepath
        except Exception as e:
            self.logger.error(f"Failed to take benchmark result snapshot: {str(e)}")
            return None
    
    def take_completion_snapshot(self):
        """Take a snapshot when benchmark is completed."""
        try:
            result_screenshot = ImageGrab.grab()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"benchmark_completed_{timestamp}.png"
            result_filepath = os.path.join(Config.BENCHMARK_RESULTS_DIR, result_filename)
            result_screenshot.save(result_filepath)
            self.logger.info(f"Final benchmark screen saved to {result_filepath}")
            return result_filepath
        except Exception as e:
            self.logger.error(f"Failed to take completion snapshot: {str(e)}")
            return None