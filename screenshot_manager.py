"""
Screenshot Manager module for capturing and managing game screenshots.
"""
import os
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from PIL import Image, ImageGrab

logger = logging.getLogger("ScreenshotManager")

class ScreenshotManager:
    """Manages screenshot capture and storage for the benchmarking process."""
    
    def __init__(self, directories: Dict[str, Path]):
        """Initialize the screenshot manager.
        
        Args:
            directories: Dictionary of directory paths for storing screenshots
        """
        self.directories = directories
        self.screenshot_counter = 0
        self.last_screenshot_time = 0
        self.last_screenshot_path = None
        
        logger.info("Screenshot Manager initialized")
    
    def take_screenshot(self, save_immediately: bool = True, custom_name: Optional[str] = None) -> Optional[str]:
        """Take a screenshot of the current screen.
        
        Args:
            save_immediately: Whether to save the screenshot immediately
            custom_name: Custom name for the screenshot file
            
        Returns:
            Path to the saved screenshot or None if failed
        """
        try:
            # Capture the screen
            screenshot = ImageGrab.grab()
            
            if save_immediately:
                # Generate filename
                if custom_name:
                    filename = f"{custom_name}.png"
                else:
                    self.screenshot_counter += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{self.screenshot_counter:04d}_{timestamp}.png"
                
                # Save to raw screenshots directory
                screenshot_path = os.path.join(self.directories["raw_screenshots"], filename)
                screenshot.save(screenshot_path)
                
                self.last_screenshot_path = screenshot_path
                self.last_screenshot_time = time.time()
                
                logger.info(f"Screenshot saved: {screenshot_path}")
                return screenshot_path
            else:
                # Return the PIL Image object without saving
                return screenshot
                
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
    
    def take_benchmark_result_screenshot(self) -> Optional[str]:
        """Take a special screenshot for benchmark results.
        
        Returns:
            Path to the saved benchmark result screenshot
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_result_{timestamp}.png"
            
            # Take screenshot with custom name in benchmark results directory
            screenshot = ImageGrab.grab()
            screenshot_path = os.path.join(self.directories["benchmark_results"], filename)
            screenshot.save(screenshot_path)
            
            logger.info(f"Benchmark result screenshot saved: {screenshot_path}")
            return screenshot_path
            
        except Exception as e:
            logger.error(f"Failed to take benchmark result screenshot: {e}")
            return None
    
    def save_screenshot_copy(self, source_path: str, destination_dir: str, 
                           custom_name: Optional[str] = None) -> Optional[str]:
        """Save a copy of an existing screenshot to a different directory.
        
        Args:
            source_path: Path to the source screenshot
            destination_dir: Destination directory
            custom_name: Custom name for the copy
            
        Returns:
            Path to the copied screenshot or None if failed
        """
        try:
            if not os.path.exists(source_path):
                logger.error(f"Source screenshot not found: {source_path}")
                return None
            
            # Generate destination filename
            if custom_name:
                filename = f"{custom_name}.png"
            else:
                source_filename = os.path.basename(source_path)
                name, ext = os.path.splitext(source_filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_copy_{timestamp}{ext}"
            
            destination_path = os.path.join(destination_dir, filename)
            
            # Copy the file
            import shutil
            shutil.copy2(source_path, destination_path)
            
            logger.info(f"Screenshot copied from {source_path} to {destination_path}")
            return destination_path
            
        except Exception as e:
            logger.error(f"Failed to copy screenshot: {e}")
            return None
    
    def take_timed_screenshot(self, delay: float = 0.0) -> Optional[str]:
        """Take a screenshot after a specified delay.
        
        Args:
            delay: Delay in seconds before taking the screenshot
            
        Returns:
            Path to the saved screenshot or None if failed
        """
        if delay > 0:
            logger.info(f"Waiting {delay} seconds before taking screenshot...")
            time.sleep(delay)
        
        return self.take_screenshot()
    
    def take_multiple_screenshots(self, count: int, interval: float = 1.0, 
                                prefix: str = "multi") -> List[str]:
        """Take multiple screenshots with specified intervals.
        
        Args:
            count: Number of screenshots to take
            interval: Interval between screenshots in seconds
            prefix: Prefix for the screenshot filenames
            
        Returns:
            List of paths to saved screenshots
        """
        screenshot_paths = []
        
        try:
            for i in range(count):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                custom_name = f"{prefix}_{i+1:02d}_{timestamp}"
                
                screenshot_path = self.take_screenshot(custom_name=custom_name)
                if screenshot_path:
                    screenshot_paths.append(screenshot_path)
                    
                    logger.info(f"Took screenshot {i+1}/{count}")
                    
                    # Wait for interval before next screenshot (except for the last one)
                    if i < count - 1:
                        time.sleep(interval)
                else:
                    logger.warning(f"Failed to take screenshot {i+1}/{count}")
            
            logger.info(f"Completed taking {len(screenshot_paths)}/{count} screenshots")
            return screenshot_paths
            
        except Exception as e:
            logger.error(f"Error taking multiple screenshots: {e}")
            return screenshot_paths
    
    def get_screenshot_info(self, screenshot_path: str) -> Optional[Dict]:
        """Get information about a screenshot file.
        
        Args:
            screenshot_path: Path to the screenshot file
            
        Returns:
            Dictionary with screenshot information or None if failed
        """
        try:
            if not os.path.exists(screenshot_path):
                return None
            
            # Get file stats
            stat = os.stat(screenshot_path)
            
            # Get image dimensions
            with Image.open(screenshot_path) as img:
                width, height = img.size
                mode = img.mode
            
            info = {
                "path": screenshot_path,
                "filename": os.path.basename(screenshot_path),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime,
                "width": width,
                "height": height,
                "mode": mode,
                "aspect_ratio": round(width / height, 2)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get screenshot info: {e}")
            return None
    
    def cleanup_old_screenshots(self, max_age_hours: float = 24.0, 
                              keep_recent_count: int = 10) -> int:
        """Clean up old screenshots to save disk space.
        
        Args:
            max_age_hours: Maximum age of screenshots to keep in hours
            keep_recent_count: Number of recent screenshots to always keep
            
        Returns:
            Number of screenshots deleted
        """
        try:
            deleted_count = 0
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Get all screenshot files
            raw_screenshots_dir = self.directories["raw_screenshots"]
            screenshot_files = []
            
            for filename in os.listdir(raw_screenshots_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(raw_screenshots_dir, filename)
                    if os.path.isfile(filepath):
                        screenshot_files.append(filepath)
            
            # Sort by modification time (newest first)
            screenshot_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Keep recent screenshots and delete old ones
            for i, filepath in enumerate(screenshot_files):
                # Always keep the most recent screenshots
                if i < keep_recent_count:
                    continue
                
                # Check age
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.debug(f"Deleted old screenshot: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to delete screenshot {filepath}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old screenshots")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during screenshot cleanup: {e}")
            return 0
    
    def get_screenshot_count(self) -> int:
        """Get the current screenshot counter value.
        
        Returns:
            Current screenshot count
        """
        return self.screenshot_counter
    
    def get_last_screenshot_path(self) -> Optional[str]:
        """Get the path to the last taken screenshot.
        
        Returns:
            Path to last screenshot or None if no screenshots taken
        """
        return self.last_screenshot_path
    
    def get_last_screenshot_time(self) -> float:
        """Get the timestamp of the last screenshot.
        
        Returns:
            Timestamp of last screenshot or 0 if no screenshots taken
        """
        return self.last_screenshot_time
    
    def time_since_last_screenshot(self) -> float:
        """Get the time elapsed since the last screenshot.
        
        Returns:
            Time in seconds since last screenshot
        """
        if self.last_screenshot_time == 0:
            return float('inf')
        return time.time() - self.last_screenshot_time
    
    def create_screenshot_summary(self) -> Dict:
        """Create a summary of all screenshots taken during the session.
        
        Returns:
            Dictionary with screenshot session summary
        """
        try:
            summary = {
                "session_start": datetime.now().isoformat(),
                "total_screenshots": self.screenshot_counter,
                "last_screenshot_time": self.last_screenshot_time,
                "directories": {key: str(path) for key, path in self.directories.items()},
                "screenshots": []
            }
            
            # Get info for all screenshots in raw_screenshots directory
            raw_screenshots_dir = self.directories["raw_screenshots"]
            if os.path.exists(raw_screenshots_dir):
                for filename in sorted(os.listdir(raw_screenshots_dir)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filepath = os.path.join(raw_screenshots_dir, filename)
                        if os.path.isfile(filepath):
                            info = self.get_screenshot_info(filepath)
                            if info:
                                summary["screenshots"].append(info)
            
            # Calculate total size
            total_size_mb = sum(info["size_mb"] for info in summary["screenshots"])
            summary["total_size_mb"] = round(total_size_mb, 2)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create screenshot summary: {e}")
            return {"error": str(e)}
    
    def save_screenshot_summary(self, output_path: Optional[str] = None) -> Optional[str]:
        """Save screenshot summary to a JSON file.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to saved summary file or None if failed
        """
        try:
            import json
            
            summary = self.create_screenshot_summary()
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_summary_{timestamp}.json"
                output_path = os.path.join(self.directories["logs"], filename)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Screenshot summary saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save screenshot summary: {e}")
            return None