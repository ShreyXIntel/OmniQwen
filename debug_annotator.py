"""
Debug Annotator Module
Handles comprehensive annotation and visualization of detected UI elements for debugging
"""

import os
import datetime
import logging
from PIL import Image, ImageDraw, ImageFont
from config import Config


class DebugAnnotator:
    def __init__(self):
        self.logger = logging.getLogger(Config.LOGGER_NAME)
        
        # Try to load a font for better text rendering
        self.font = self._load_font()
        self.font_small = self._load_font(size=Config.ANNOTATION_FONT_SIZE_SMALL)
        
    def _load_font(self, size=None):
        """Load a font for text rendering, with fallback to default."""
        size = size or Config.ANNOTATION_FONT_SIZE
        
        try:
            # Try to load a system font
            if os.name == 'nt':  # Windows
                font_paths = [
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/calibri.ttf",
                    "C:/Windows/Fonts/tahoma.ttf"
                ]
            else:  # Linux/Mac
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/System/Library/Fonts/Arial.ttf",
                    "/usr/share/fonts/TTF/arial.ttf"
                ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, size)
            
            # Fallback to default font
            return ImageFont.load_default()
            
        except Exception as e:
            self.logger.warning(f"Could not load custom font: {e}, using default")
            return ImageFont.load_default()
    
    def annotate_image(self, image_path, parsed_content, save_path=None):
        """
        Create a comprehensive annotated image with all detected elements.
        
        Args:
            image_path: Path to the original image
            parsed_content: List of detected elements with bboxes and metadata
            save_path: Optional custom save path, otherwise auto-generated
            
        Returns:
            Path to the saved annotated image
        """
        try:
            # Load the original image
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Create annotations for each element
            for i, element in enumerate(parsed_content):
                if not isinstance(element, dict) or 'bbox' not in element:
                    continue
                
                # Extract element information
                bbox = element.get('bbox', [])
                content = element.get('content', f'Element_{i}')
                interactivity = element.get('interactivity', False)
                element_type = element.get('type', 'unknown')
                confidence = element.get('confidence', 0.0)
                
                if len(bbox) != 4:
                    continue
                
                # Convert ratio coordinates to pixel coordinates
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = bbox
                x1 = int(x1_ratio * img_width)
                y1 = int(y1_ratio * img_height)
                x2 = int(x2_ratio * img_width)
                y2 = int(y2_ratio * img_height)
                
                # Choose colors based on element properties
                colors = self._get_element_colors(interactivity, element_type)
                
                # Draw the bounding box
                self._draw_bounding_box(
                    draw, x1, y1, x2, y2, colors, i, 
                    content, interactivity, confidence
                )
            
            # Add legend to the image
            self._add_legend(draw, img_width, img_height)
            
            # Add summary information
            self._add_summary(draw, parsed_content, img_width, img_height)
            
            # Generate save path if not provided
            if save_path is None:
                save_path = self._generate_save_path()
            
            # Save the annotated image
            image.save(save_path)
            self.logger.info(f"Debug annotated image saved to: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to create annotated image: {str(e)}")
            return None
    
    def _get_element_colors(self, interactivity, element_type):
        """Get colors for an element based on its properties."""
        if interactivity:
            # Interactive elements get bright colors
            box_color = Config.ANNOTATION_COLOR_INTERACTIVE
            text_bg_color = Config.ANNOTATION_COLOR_INTERACTIVE_BG
        else:
            # Non-interactive elements get muted colors
            box_color = Config.ANNOTATION_COLOR_NON_INTERACTIVE
            text_bg_color = Config.ANNOTATION_COLOR_NON_INTERACTIVE_BG
        
        return {
            'box': box_color,
            'text_bg': text_bg_color,
            'text': Config.ANNOTATION_COLOR_TEXT
        }
    
    def _draw_bounding_box(self, draw, x1, y1, x2, y2, colors, index, content, interactivity, confidence):
        """Draw a single bounding box with annotations."""
        # Draw the main bounding box
        draw.rectangle(
            [x1, y1, x2, y2], 
            outline=colors['box'], 
            width=Config.ANNOTATION_BOX_WIDTH
        )
        
        # Prepare the label text
        interactive_marker = "🔗" if interactivity else "📄"
        label = f"{index+1}: {interactive_marker} {content[:Config.ANNOTATION_MAX_TEXT_LENGTH]}"
        if len(content) > Config.ANNOTATION_MAX_TEXT_LENGTH:
            label += "..."
        
        # Add confidence if available
        if confidence > 0:
            label += f" ({confidence:.2f})"
        
        # Calculate text position and size
        text_bbox = draw.textbbox((0, 0), label, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Position the label above the box, or below if there's no room above
        label_x = x1
        label_y = y1 - text_height - Config.ANNOTATION_TEXT_PADDING
        
        if label_y < 0:  # Not enough room above, put it below
            label_y = y2 + Config.ANNOTATION_TEXT_PADDING
        
        # Draw text background
        bg_x1 = label_x - Config.ANNOTATION_TEXT_PADDING
        bg_y1 = label_y - Config.ANNOTATION_TEXT_PADDING
        bg_x2 = label_x + text_width + Config.ANNOTATION_TEXT_PADDING
        bg_y2 = label_y + text_height + Config.ANNOTATION_TEXT_PADDING
        
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=colors['text_bg'])
        
        # Draw the text
        draw.text((label_x, label_y), label, fill=colors['text'], font=self.font)
        
        # Draw element index in the top-left corner of the box
        index_text = str(index + 1)
        index_bbox = draw.textbbox((0, 0), index_text, font=self.font_small)
        index_width = index_bbox[2] - index_bbox[0]
        index_height = index_bbox[3] - index_bbox[1]
        
        # Index background
        idx_bg_x1 = x1
        idx_bg_y1 = y1
        idx_bg_x2 = x1 + index_width + 4
        idx_bg_y2 = y1 + index_height + 4
        
        draw.rectangle([idx_bg_x1, idx_bg_y1, idx_bg_x2, idx_bg_y2], fill=colors['box'])
        draw.text((x1 + 2, y1 + 2), index_text, fill='white', font=self.font_small)
    
    def _add_legend(self, draw, img_width, img_height):
        """Add a legend explaining the color coding."""
        legend_items = [
            ("🔗 Interactive Element", Config.ANNOTATION_COLOR_INTERACTIVE),
            ("📄 Non-Interactive Element", Config.ANNOTATION_COLOR_NON_INTERACTIVE),
        ]
        
        # Position legend in the top-right corner
        legend_x = img_width - Config.ANNOTATION_LEGEND_WIDTH
        legend_y = Config.ANNOTATION_LEGEND_MARGIN
        
        # Draw legend background
        legend_bg_x2 = img_width - Config.ANNOTATION_LEGEND_MARGIN
        legend_bg_y2 = legend_y + len(legend_items) * Config.ANNOTATION_LEGEND_LINE_HEIGHT + Config.ANNOTATION_LEGEND_MARGIN
        
        draw.rectangle(
            [legend_x, legend_y, legend_bg_x2, legend_bg_y2],
            fill=Config.ANNOTATION_LEGEND_BG,
            outline=Config.ANNOTATION_LEGEND_BORDER,
            width=2
        )
        
        # Draw legend title
        title_y = legend_y + Config.ANNOTATION_LEGEND_MARGIN // 2
        draw.text((legend_x + 10, title_y), "Legend:", fill=Config.ANNOTATION_COLOR_TEXT, font=self.font)
        
        # Draw legend items
        for i, (text, color) in enumerate(legend_items):
            item_y = title_y + Config.ANNOTATION_LEGEND_LINE_HEIGHT + (i * Config.ANNOTATION_LEGEND_LINE_HEIGHT)
            
            # Draw color box
            color_box_x1 = legend_x + 10
            color_box_y1 = item_y + 2
            color_box_x2 = color_box_x1 + 15
            color_box_y2 = item_y + 12
            
            draw.rectangle([color_box_x1, color_box_y1, color_box_x2, color_box_y2], fill=color)
            
            # Draw text
            draw.text((color_box_x2 + 5, item_y), text, fill=Config.ANNOTATION_COLOR_TEXT, font=self.font_small)
    
    def _add_summary(self, draw, parsed_content, img_width, img_height):
        """Add a summary of detected elements."""
        if not parsed_content:
            return
        
        # Count elements by type
        total_elements = len(parsed_content)
        interactive_count = sum(1 for el in parsed_content 
                              if isinstance(el, dict) and el.get('interactivity', False))
        non_interactive_count = total_elements - interactive_count
        
        # Create summary text
        summary_lines = [
            f"Detection Summary:",
            f"Total Elements: {total_elements}",
            f"Interactive: {interactive_count}",
            f"Non-Interactive: {non_interactive_count}",
            f"Timestamp: {datetime.datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Position summary in bottom-left corner
        summary_x = Config.ANNOTATION_LEGEND_MARGIN
        line_height = Config.ANNOTATION_LEGEND_LINE_HEIGHT
        summary_height = len(summary_lines) * line_height + Config.ANNOTATION_LEGEND_MARGIN
        summary_y = img_height - summary_height - Config.ANNOTATION_LEGEND_MARGIN
        
        # Calculate summary box width
        max_width = max(draw.textbbox((0, 0), line, font=self.font_small)[2] 
                       for line in summary_lines)
        summary_width = max_width + 2 * Config.ANNOTATION_LEGEND_MARGIN
        
        # Draw summary background
        draw.rectangle(
            [summary_x, summary_y, summary_x + summary_width, img_height - Config.ANNOTATION_LEGEND_MARGIN],
            fill=Config.ANNOTATION_LEGEND_BG,
            outline=Config.ANNOTATION_LEGEND_BORDER,
            width=2
        )
        
        # Draw summary text
        for i, line in enumerate(summary_lines):
            text_y = summary_y + Config.ANNOTATION_LEGEND_MARGIN // 2 + (i * line_height)
            font_to_use = self.font if i == 0 else self.font_small  # Title uses larger font
            draw.text((summary_x + 10, text_y), line, fill=Config.ANNOTATION_COLOR_TEXT, font=font_to_use)
    
    def _generate_save_path(self):
        """Generate a save path for the annotated image."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_annotated_{timestamp}.png"
        return os.path.join(Config.PARSED_IMAGES_DIR, filename)
    
    def create_comparison_image(self, original_path, annotated_path, save_path=None):
        """Create a side-by-side comparison of original and annotated images."""
        try:
            # Load both images
            original = Image.open(original_path).convert("RGB")
            annotated = Image.open(annotated_path).convert("RGB")
            
            # Resize if needed to match dimensions
            if original.size != annotated.size:
                # Resize annotated to match original
                annotated = annotated.resize(original.size, Image.Resampling.LANCZOS)
            
            # Create side-by-side image
            width, height = original.size
            comparison = Image.new('RGB', (width * 2 + Config.ANNOTATION_COMPARISON_GAP, height), 'white')
            
            # Paste images
            comparison.paste(original, (0, 0))
            comparison.paste(annotated, (width + Config.ANNOTATION_COMPARISON_GAP, 0))
            
            # Add labels
            draw = ImageDraw.Draw(comparison)
            draw.text((10, 10), "Original", fill='black', font=self.font)
            draw.text((width + Config.ANNOTATION_COMPARISON_GAP + 10, 10), "Annotated", fill='black', font=self.font)
            
            # Generate save path if not provided
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparison_{timestamp}.png"
                save_path = os.path.join(Config.PARSED_IMAGES_DIR, filename)
            
            # Save comparison image
            comparison.save(save_path)
            self.logger.info(f"Comparison image saved to: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison image: {str(e)}")
            return None
    
    def annotate_elements_list(self, parsed_content, save_path=None):
        """Create a text file with detailed element information."""
        try:
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"elements_list_{timestamp}.txt"
                save_path = os.path.join(Config.PARSED_IMAGES_DIR, filename)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"UI Elements Detection Report\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Elements: {len(parsed_content)}\n")
                f.write("="*80 + "\n\n")
                
                for i, element in enumerate(parsed_content):
                    if not isinstance(element, dict):
                        continue
                    
                    f.write(f"Element #{i+1}\n")
                    f.write(f"  Content: {element.get('content', 'N/A')}\n")
                    f.write(f"  Interactive: {element.get('interactivity', False)}\n")
                    f.write(f"  Type: {element.get('type', 'unknown')}\n")
                    f.write(f"  Confidence: {element.get('confidence', 'N/A')}\n")
                    f.write(f"  BBox (ratio): {element.get('bbox', 'N/A')}\n")
                    
                    # Add any additional metadata
                    for key, value in element.items():
                        if key not in ['content', 'interactivity', 'type', 'confidence', 'bbox']:
                            f.write(f"  {key}: {value}\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Elements list saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to create elements list: {str(e)}")
            return None