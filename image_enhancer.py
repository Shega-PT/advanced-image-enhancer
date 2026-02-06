"""
Advanced Image Enhancer - Professional-grade image processing pipeline.
Enhances visual quality while preserving natural appearance and avoiding artifacts.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import cv2
import numpy as np
from PIL import Image

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class EnhancementMode(Enum):
    """Enhancement modes for different types of images."""
    NATURAL = "natural"          # Preserve natural look
    SHARP = "sharp"              # Emphasize sharpness
    VIBRANT = "vibrant"          # Boost colors and contrast
    PORTRAIT = "portrait"        # Optimized for portraits
    LANDSCAPE = "landscape"      # Optimized for landscapes


class ProcessingStep(Enum):
    """Processing steps in the enhancement pipeline."""
    LOADING = "loading"
    RESIZING = "resizing"
    DENOISING = "denoising"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    SHARPENING = "sharpening"
    SAVING = "saving"


@dataclass
class QualityMetrics:
    """Quality metrics for image analysis."""
    sharpness: float
    contrast: float
    snr: float  # Signal-to-Noise Ratio
    brightness: float
    color_variance: float
    
    @classmethod
    def from_images(cls, before: np.ndarray, after: np.ndarray) -> 'QualityMetrics':
        """Calculate metrics from before/after images."""
        # Convert to grayscale for some metrics
        if len(before.shape) == 3:
            gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        else:
            gray_before = before
            gray_after = after
        
        # Calculate metrics
        sharpness_before = cv2.Laplacian(gray_before, cv2.CV_64F).var()
        sharpness_after = cv2.Laplacian(gray_after, cv2.CV_64F).var()
        
        contrast_before = np.std(gray_before)
        contrast_after = np.std(gray_after)
        
        snr_before = np.mean(gray_before) / np.std(gray_before) if np.std(gray_before) > 0 else 0
        snr_after = np.mean(gray_after) / np.std(gray_after) if np.std(gray_after) > 0 else 0
        
        brightness_before = np.mean(gray_before)
        brightness_after = np.mean(gray_after)
        
        # Color variance (for color images only)
        if len(before.shape) == 3:
            color_var_before = np.mean([np.std(before[:, :, i]) for i in range(3)])
            color_var_after = np.mean([np.std(after[:, :, i]) for i in range(3)])
        else:
            color_var_before = 0
            color_var_after = 0
        
        return cls(
            sharpness=sharpness_after - sharpness_before,
            contrast=contrast_after - contrast_before,
            snr=snr_after - snr_before,
            brightness=brightness_after - brightness_before,
            color_variance=color_var_after - color_var_before
        )


@dataclass
class ProcessingResult:
    """Results of processing a single image."""
    input_path: Path
    output_path: Path
    success: bool
    processing_time: float
    quality_metrics: Optional[QualityMetrics] = None
    file_size_change: float = 0.0
    error_message: Optional[str] = None
    step_times: Dict[str, float] = None
    
    def __post_init__(self):
        if self.step_times is None:
            self.step_times = {}


@dataclass
class EnhancementConfig:
    """Configuration for image enhancement."""
    target_size: Tuple[int, int] = (1920, 1080)
    mode: EnhancementMode = EnhancementMode.NATURAL
    denoise_strength: float = 3.0
    sharpening_strength: float = 1.2
    contrast_boost: float = 1.1
    saturation_boost: float = 1.1
    preserve_original: bool = True
    
    def get_mode_params(self) -> Dict[str, Any]:
        """Get parameters based on enhancement mode."""
        params = {
            "denoise_strength": self.denoise_strength,
            "sharpening_strength": self.sharpening_strength,
            "contrast_boost": self.contrast_boost,
            "saturation_boost": self.saturation_boost,
        }
        
        # Mode-specific adjustments
        if self.mode == EnhancementMode.SHARP:
            params.update({
                "sharpening_strength": 1.5,
                "denoise_strength": 2.0,
            })
        elif self.mode == EnhancementMode.VIBRANT:
            params.update({
                "contrast_boost": 1.3,
                "saturation_boost": 1.3,
            })
        elif self.mode == EnhancementMode.PORTRAIT:
            params.update({
                "denoise_strength": 4.0,
                "sharpening_strength": 1.1,
                "contrast_boost": 1.05,
            })
        elif self.mode == EnhancementMode.LANDSCAPE:
            params.update({
                "sharpening_strength": 1.4,
                "saturation_boost": 1.2,
            })
        
        return params


# ============================================================================
# IMAGE ENHANCER
# ============================================================================

class AdvancedImageEnhancer:
    """
    Advanced image enhancement pipeline with professional-grade processing.
    Enhances visual quality while maintaining natural appearance.
    """
    
    def __init__(self, config: EnhancementConfig = None):
        """
        Initialize the image enhancer.
        
        Args:
            config: Enhancement configuration (uses default if None)
        """
        self.config = config or EnhancementConfig()
        self.logger = logging.getLogger(__name__)
    
    def _start_timer(self, step: ProcessingStep) -> float:
        """Start timing a processing step."""
        return time.time()
    
    def _end_timer(self, start_time: float, step: ProcessingStep, 
                  step_times: Dict[str, float]) -> float:
        """End timing a processing step and store duration."""
        duration = time.time() - start_time
        step_times[step.value] = duration
        return duration
    
    def _calculate_metrics(self, original: np.ndarray, 
                          enhanced: np.ndarray) -> QualityMetrics:
        """Calculate quality improvement metrics."""
        return QualityMetrics.from_images(original, enhanced)
    
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image with validation."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB for consistency
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        
        except Exception as e:
            self.logger.error(f"Error loading {image_path}: {e}")
            return None
    
    def _resize_to_target(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions while maintaining quality.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        target_width, target_height = self.config.target_size
        
        # Only resize if needed
        if (width, height) == (target_width, target_height):
            return image.copy()
        
        # Choose interpolation method based on scaling direction
        if width < target_width or height < target_height:
            # Upscaling - use Lanczos for quality
            interpolation = cv2.INTER_LANCZOS4
        else:
            # Downscaling - use area averaging for quality
            interpolation = cv2.INTER_AREA
        
        return cv2.resize(image, (target_width, target_height), 
                         interpolation=interpolation)
    
    def _apply_non_local_means_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Apply non-local means denoising to reduce noise while preserving details.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        params = self.config.get_mode_params()
        strength = params["denoise_strength"]
        
        # Convert to LAB color space for better denoising
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply denoising to luminance channel only
        l_denoised = cv2.fastNlMeansDenoising(
            l, None,
            h=strength * 10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Merge channels back
        lab_denoised = cv2.merge([l_denoised, a, b])
        denoised = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2RGB)
        
        return denoised
    
    def _enhance_contrast_local(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        params = self.config.get_mode_params()
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to luminance channel
        clahe = cv2.createCLAHE(
            clipLimit=params["contrast_boost"] * 2,
            tileGridSize=(8, 8)
        )
        l_enhanced = clahe.apply(l)
        
        # Apply bilateral filter for smoothness
        l_enhanced = cv2.bilateralFilter(l_enhanced, 5, 75, 75)
        
        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _enhance_saturation(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance color saturation.
        
        Args:
            image: Input image
            
        Returns:
            Saturation-enhanced image
        """
        params = self.config.get_mode_params()
        saturation_factor = params["saturation_boost"]
        
        if saturation_factor == 1.0:
            return image
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Enhance saturation channel
        s = np.clip(s.astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        return enhanced
    
    def _adaptive_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive sharpening that varies based on edge strength.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        params = self.config.get_mode_params()
        strength = params["sharpening_strength"]
        
        if strength == 1.0:
            return image
        
        # Convert to grayscale for edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Create edge mask with Gaussian blur for smooth transitions
        edge_mask = cv2.dilate(edges, None, iterations=1)
        edge_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (5, 5), 1.0)
        edge_mask = np.clip(edge_mask / 255.0, 0, 1)
        
        # Expand mask to 3 channels if needed
        if len(image.shape) == 3:
            edge_mask = np.stack([edge_mask] * 3, axis=2)
        
        # Create sharpened version using unsharp masking
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        
        # Blend based on edge mask (only sharpen edges)
        result = image * (1.0 - edge_mask) + sharpened * edge_mask
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _apply_super_resolution_effect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply super-resolution-like enhancement.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Step 1: Initial upscaling for detail processing
        scale_factor = 2
        upscaled = cv2.resize(
            image,
            (image.shape[1] * scale_factor, image.shape[0] * scale_factor),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Step 2: Process at higher resolution
        upscaled = self._enhance_contrast_local(upscaled)
        upscaled = self._enhance_saturation(upscaled)
        
        # Step 3: Downscale with area interpolation for quality
        downscaled = cv2.resize(
            upscaled,
            self.config.target_size,
            interpolation=cv2.INTER_AREA
        )
        
        return downscaled
    
    def _save_image(self, image: np.ndarray, output_path: Path) -> bool:
        """
        Save image with appropriate format and quality.
        
        Args:
            image: Image to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert RGB back to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Determine save parameters based on file extension
            extension = output_path.suffix.lower()
            
            if extension in ['.jpg', '.jpeg']:
                # JPEG with quality setting
                cv2.imwrite(
                    str(output_path),
                    image_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
            elif extension == '.png':
                # PNG with compression
                cv2.imwrite(
                    str(output_path),
                    image_bgr,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3]
                )
            else:
                # Default save
                cv2.imwrite(str(output_path), image_bgr)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save {output_path}: {e}")
            return False
    
    def enhance_image(self, 
                     input_path: Path,
                     output_path: Optional[Path] = None) -> ProcessingResult:
        """
        Main enhancement pipeline for a single image.
        
        Args:
            input_path: Path to input image
            output_path: Path for output image (auto-generated if None)
            
        Returns:
            Processing result
        """
        start_time = time.time()
        step_times = {}
        
        try:
            # Step 1: Load image
            step_start = self._start_timer(ProcessingStep.LOADING)
            original = self._load_image(input_path)
            if original is None:
                raise ValueError(f"Failed to load image: {input_path}")
            self._end_timer(step_start, ProcessingStep.LOADING, step_times)
            
            # Generate output path if not provided
            if output_path is None:
                output_path = self._generate_output_path(input_path)
            
            # Store original file size
            original_size_kb = input_path.stat().st_size / 1024
            
            # Step 2: Resize to target dimensions
            step_start = self._start_timer(ProcessingStep.RESIZING)
            resized = self._resize_to_target(original)
            self._end_timer(step_start, ProcessingStep.RESIZING, step_times)
            
            # Step 3: Apply denoising
            step_start = self._start_timer(ProcessingStep.DENOISING)
            denoised = self._apply_non_local_means_denoising(resized)
            self._end_timer(step_start, ProcessingStep.DENOISING, step_times)
            
            # Step 4: Apply super-resolution effect
            enhanced = denoised.copy()
            
            # Only apply super-resolution for appropriate modes
            if self.config.mode in [EnhancementMode.SHARP, EnhancementMode.LANDSCAPE]:
                enhanced = self._apply_super_resolution_effect(enhanced)
            
            # Step 5: Contrast enhancement
            step_start = self._start_timer(ProcessingStep.CONTRAST_ENHANCEMENT)
            enhanced = self._enhance_contrast_local(enhanced)
            enhanced = self._enhance_saturation(enhanced)
            self._end_timer(step_start, ProcessingStep.CONTRAST_ENHANCEMENT, step_times)
            
            # Step 6: Adaptive sharpening
            step_start = self._start_timer(ProcessingStep.SHARPENING)
            enhanced = self._adaptive_sharpening(enhanced)
            self._end_timer(step_start, ProcessingStep.SHARPENING, step_times)
            
            # Step 7: Save image
            step_start = self._start_timer(ProcessingStep.SAVING)
            save_success = self._save_image(enhanced, output_path)
            self._end_timer(step_start, ProcessingStep.SAVING, step_times)
            
            if not save_success:
                raise RuntimeError(f"Failed to save image to {output_path}")
            
            # Calculate metrics
            processing_time = time.time() - start_time
            metrics = self._calculate_metrics(resized, enhanced)
            
            # Calculate file size change
            output_size_kb = output_path.stat().st_size / 1024
            file_size_change = ((output_size_kb - original_size_kb) / 
                               original_size_kb * 100)
            
            # Create result
            result = ProcessingResult(
                input_path=input_path,
                output_path=output_path,
                success=True,
                processing_time=processing_time,
                quality_metrics=metrics,
                file_size_change=file_size_change,
                step_times=step_times
            )
            
            self._log_success(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to process {input_path}: {e}")
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output_path or Path(),
                success=False,
                processing_time=processing_time,
                error_message=str(e),
                step_times=step_times
            )
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """
        Generate output path based on input path and configuration.
        
        Args:
            input_path: Input file path
            
        Returns:
            Output file path
        """
        # Create output directory
        output_dir = input_path.parent / "enhanced"
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        stem = input_path.stem
        suffix = f"_enhanced_{self.config.mode.value}.png"
        
        return output_dir / f"{stem}{suffix}"
    
    def _log_success(self, result: ProcessingResult):
        """Log successful processing results."""
        metrics = result.quality_metrics
        
        self.logger.info(f"✓ Success: {result.input_path.name}")
        self.logger.info(f"  Output: {result.output_path.name}")
        self.logger.info(f"  Time: {result.processing_time:.2f}s")
        
        if metrics:
            self.logger.info(f"  Sharpness improvement: {metrics.sharpness:+.1f}")
            self.logger.info(f"  Contrast improvement: {metrics.contrast:+.1f}")
            self.logger.info(f"  SNR improvement: {metrics.snr:+.2f}")
        
        self.logger.info(f"  File size change: {result.file_size_change:+.1f}%")
        
        # Log step times if verbose
        if self.logger.isEnabledFor(logging.DEBUG):
            for step, duration in result.step_times.items():
                self.logger.debug(f"    {step}: {duration:.2f}s")
    
    def process_batch(self, 
                     input_dir: Path,
                     patterns: List[str] = None) -> List[ProcessingResult]:
        """
        Process all matching images in a directory.
        
        Args:
            input_dir: Directory containing images
            patterns: File patterns to match
            
        Returns:
            List of processing results
        """
        if patterns is None:
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        # Find all matching files
        files = []
        for pattern in patterns:
            files.extend(input_dir.glob(pattern))
        
        if not files:
            self.logger.warning(f"No matching files found in {input_dir}")
            return []
        
        self.logger.info(f"Starting batch processing of {len(files)} images")
        self.logger.info(f"Mode: {self.config.mode.value}")
        self.logger.info(f"Target size: {self.config.target_size[0]}x{self.config.target_size[1]}")
        
        results = []
        total_start_time = time.time()
        
        for idx, file_path in enumerate(files, 1):
            self.logger.info(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")
            
            result = self.enhance_image(file_path)
            results.append(result)
            
            if not result.success:
                self.logger.error(f"  Failed: {result.error_message}")
        
        # Calculate batch statistics
        total_time = time.time() - total_start_time
        successful = [r for r in results if r.success]
        
        if successful:
            avg_time = sum(r.processing_time for r in successful) / len(successful)
            avg_sharpness = sum(r.quality_metrics.sharpness for r in successful if r.quality_metrics) / len(successful)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info("BATCH PROCESSING SUMMARY")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Total files: {len(files)}")
            self.logger.info(f"Successfully processed: {len(successful)}")
            self.logger.info(f"Failed: {len(files) - len(successful)}")
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Average time per image: {avg_time:.2f}s")
            self.logger.info(f"Average sharpness improvement: {avg_sharpness:+.1f}")
        
        return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced Image Enhancer - Professional image quality enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg                           # Enhance single image
  %(prog)s --input-dir ./photos --batch        # Batch process directory
  %(prog)s --mode vibrant --size 4k            # Vibrant mode at 4K
  %(prog)s --mode portrait --denoise 4.0       # Portrait with strong denoising
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="Input image file (optional if using --input-dir)"
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        default=".",
        help="Input directory for batch processing"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file or directory"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=[m.value for m in EnhancementMode],
        default="natural",
        help="Enhancement mode (default: natural)"
    )
    
    parser.add_argument(
        "--size", "-s",
        default="1080p",
        choices=["720p", "1080p", "1440p", "4k", "original"],
        help="Target resolution (default: 1080p)"
    )
    
    parser.add_argument(
        "--denoise", "-d",
        type=float,
        default=3.0,
        help="Denoising strength (1.0-10.0, default: 3.0)"
    )
    
    parser.add_argument(
        "--sharpen", "-sh",
        type=float,
        default=1.2,
        help="Sharpening strength (1.0-2.0, default: 1.2)"
    )
    
    parser.add_argument(
        "--contrast", "-c",
        type=float,
        default=1.1,
        help="Contrast boost (1.0-2.0, default: 1.1)"
    )
    
    parser.add_argument(
        "--saturation", "-sat",
        type=float,
        default=1.1,
        help="Saturation boost (1.0-2.0, default: 1.1)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch process all images in input directory"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--preserve-original",
        action="store_true",
        default=True,
        help="Preserve original files (default: True)"
    )
    
    return parser.parse_args()


def get_target_size(size_str: str, original_size: Tuple[int, int] = None) -> Tuple[int, int]:
    """Convert resolution string to (width, height) tuple."""
    resolutions = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4k": (3840, 2160)
    }
    
    if size_str == "original" and original_size:
        return original_size
    else:
        return resolutions.get(size_str.lower(), (1920, 1080))


def setup_logging(verbose: bool):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('image_enhancer.log', mode='w', encoding='utf-8')
        ]
    )


def print_welcome_banner():
    """Print welcome banner."""
    print(f"""
{'#'*70}
{'🎨 ADVANCED IMAGE ENHANCER 🎨'.center(70)}
{'#'*70}
Professional-grade image processing pipeline
Enhances visual quality while preserving natural appearance
    """)


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        print_welcome_banner()
        
        # Create configuration
        mode = EnhancementMode(args.mode)
        
        # Create config with user parameters
        config = EnhancementConfig(
            mode=mode,
            denoise_strength=args.denoise,
            sharpening_strength=args.sharpen,
            contrast_boost=args.contrast,
            saturation_boost=args.saturation,
            preserve_original=args.preserve_original
        )
        
        # Create enhancer
        enhancer = AdvancedImageEnhancer(config)
        
        if args.batch or args.input is None:
            # Batch processing
            input_dir = Path(args.input_dir).resolve()
            
            if not input_dir.exists():
                logger.error(f"Input directory does not exist: {input_dir}")
                sys.exit(1)
            
            logger.info(f"Batch processing directory: {input_dir}")
            logger.info(f"Enhancement mode: {mode.value}")
            logger.info(f"Denoising strength: {args.denoise}")
            logger.info(f"Sharpening strength: {args.sharpen}")
            
            # Process batch
            results = enhancer.process_batch(input_dir)
            
            # Save summary report
            if results:
                successful = [r for r in results if r.success]
                if successful:
                    # Create summary
                    summary = {
                        'total_processed': len(results),
                        'successful': len(successful),
                        'failed': len(results) - len(successful),
                        'enhancement_mode': mode.value,
                        'timestamp': datetime.now().isoformat(),
                        'results': [
                            {
                                'input': str(r.input_path),
                                'output': str(r.output_path),
                                'success': r.success,
                                'processing_time': r.processing_time,
                                'file_size_change': r.file_size_change,
                                'sharpness_improvement': r.quality_metrics.sharpness if r.quality_metrics else 0
                            }
                            for r in successful
                        ]
                    }
                    
                    # Save to JSON
                    try:
                        import json
                        report_path = input_dir / "enhancement_report.json"
                        with open(report_path, 'w') as f:
                            json.dump(summary, f, indent=2, default=str)
                        logger.info(f"Report saved to: {report_path}")
                    except Exception as e:
                        logger.warning(f"Could not save report: {e}")
            
        else:
            # Single file processing
            input_path = Path(args.input).resolve()
            
            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_path}")
                sys.exit(1)
            
            # Determine target size
            if args.size == "original":
                # Load image to get original size
                image = cv2.imread(str(input_path))
                if image is not None:
                    target_size = (image.shape[1], image.shape[0])
                    config.target_size = target_size
            else:
                target_size = get_target_size(args.size)
                config.target_size = target_size
            
            # Determine output path
            if args.output:
                output_path = Path(args.output).resolve()
            else:
                output_path = None
            
            logger.info(f"Processing: {input_path.name}")
            logger.info(f"Enhancement mode: {mode.value}")
            logger.info(f"Target size: {config.target_size[0]}x{config.target_size[1]}")
            
            # Process image
            result = enhancer.enhance_image(input_path, output_path)
            
            # Print result
            print(f"\n{'='*60}")
            if result.success:
                print(f"✅ ENHANCEMENT COMPLETE")
                print(f"{'-'*60}")
                print(f"Input:  {result.input_path.name}")
                print(f"Output: {result.output_path.name}")
                print(f"Time:   {result.processing_time:.2f}s")
                
                if result.quality_metrics:
                    print(f"\n📊 QUALITY IMPROVEMENT:")
                    print(f"  • Sharpness:    {result.quality_metrics.sharpness:+.1f}")
                    print(f"  • Contrast:     {result.quality_metrics.contrast:+.1f}")
                    print(f"  • Color Vibrancy: {result.quality_metrics.color_variance:+.1f}")
                
                print(f"\n💾 FILE SIZE:")
                orig_kb = result.input_path.stat().st_size / 1024
                enh_kb = result.output_path.stat().st_size / 1024
                print(f"  Original: {orig_kb:.1f} KB")
                print(f"  Enhanced: {enh_kb:.1f} KB")
                print(f"  Change:   {result.file_size_change:+.1f}%")
                
                print(f"\n⚙️  PROCESSING STEPS:")
                for step, duration in result.step_times.items():
                    print(f"  • {step}: {duration:.2f}s")
                
                print(f"\n🎯 ENHANCEMENT MODE: {mode.value.upper()}")
                
            else:
                print(f"❌ ENHANCEMENT FAILED")
                print(f"{'-'*60}")
                print(f"Error: {result.error_message}")
            
            print(f"{'='*60}")
        
        return 0
    
    except KeyboardInterrupt:
        print(f"\n\n{'🛑 OPERATION INTERRUPTED'.center(60)}")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
