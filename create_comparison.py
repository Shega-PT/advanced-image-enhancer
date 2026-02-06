## Bonus File ##
"""
Create visual comparison between original and enhanced images.
"""

import cv2
import numpy as np
from pathlib import Path

def create_comparison_grid(original_path: Path, enhanced_path: Path, 
                          output_path: Path, labels: tuple = ("Original", "Enhanced")):
    """
    Create a comparison grid with labels and metrics.
    
    Args:
        original_path: Path to original image
        enhanced_path: Path to enhanced image
        output_path: Path for output comparison
        labels: Tuple of (original_label, enhanced_label)
    """
    # Load images
    original = cv2.imread(str(original_path))
    enhanced = cv2.imread(str(enhanced_path))
    
    if original is None or enhanced is None:
        print("Error loading images")
        return
    
    # Resize to same height
    height = min(original.shape[0], enhanced.shape[0])
    width1 = int(original.shape[1] * height / original.shape[0])
    width2 = int(enhanced.shape[1] * height / enhanced.shape[0])
    
    original = cv2.resize(original, (width1, height))
    enhanced = cv2.resize(enhanced, (width2, height))
    
    # Create side-by-side comparison
    comparison = np.hstack([original, enhanced])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (255, 255, 255)
    
    # Original label
    cv2.putText(comparison, labels[0], 
               (20, 50), font, font_scale, color, thickness)
    
    # Enhanced label
    cv2.putText(comparison, labels[1], 
               (width1 + 20, 50), font, font_scale, color, thickness)
    
    # Add divider
    cv2.line(comparison, (width1, 0), (width1, height), color, 2)
    
    # Save
    cv2.imwrite(str(output_path), comparison)
    print(f"Comparison saved: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python create_comparison.py original.jpg enhanced.jpg comparison.jpg")
        sys.exit(1)
    
    create_comparison_grid(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
