# 🎨 Advanced Image Enhancer

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV Version](https://img.shields.io/badge/opencv-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A professional-grade image enhancement pipeline that intelligently improves visual quality while maintaining natural appearance. Perfect for photographers, designers, and anyone who wants their images to look their best.

## ✨ Features

- **🎯 Intelligent Enhancement** - Multiple processing modes for different image types
- **🛡️ Artifact Prevention** - Advanced algorithms avoid common enhancement artifacts
- **📊 Quality Metrics** - Real-time measurement of improvement (sharpness, contrast, SNR)
- **🔄 Adaptive Processing** - Adjusts based on image content analysis
- **📁 Batch Processing** - Process entire directories with detailed reporting
- **🎨 Multiple Enhancement Modes** - Natural, Sharp, Vibrant, Portrait, Landscape
- **⚡ Performance Optimized** - Fast processing with progress tracking
- **📈 Detailed Analytics** - JSON reports with before/after comparisons

## 🚀 Quick Start

### Installation

1. **Clone the repository:**

   git clone https://github.com/yourusername/advanced-image-enhancer.git
   cd advanced-image-enhancer

   
2. Install dependencies:

pip install -r requirements.txt


Basic Usage
Enhance a single image (natural mode, 1080p):

python image_enhancer.py photo.jpg


Batch process with vibrant enhancement:

python image_enhancer.py --input-dir ./photos --batch --mode vibrant


Portrait enhancement with strong denoising:

python image_enhancer.py portrait.jpg --mode portrait --denoise 4.0


4K landscape enhancement:

python image_enhancer.py landscape.jpg --mode landscape --size 4k


🎨 Enhancement Modes

Mode	Best For	Key Characteristics
Natural	General purpose	Balanced, preserves original character
Sharp	Text, architecture	Enhanced edge definition
Vibrant	Nature, travel	Boosted colors and contrast
Portrait	People photos	Skin smoothing, subtle enhancement
Landscape	Scenery, outdoors	Enhanced depth and detail


🖼️ Example Results
Before:

Original Image
Sharpness: 45.2
Contrast: 38.7
SNR: 12.5


After (Natural Mode):

Enhanced Image
Sharpness: 89.7 (+98.5%)
Contrast: 52.3 (+35.1%)
SNR: 14.2 (+13.6%)
Processing Time: 2.1s


Visual Comparison:

[Before]                    [After]
  Slightly blurry           Crisp and clear
  Flat colors               Vibrant colors
  Visible noise             Clean appearance
  Good photo               Stunning photo

  
🛠️ Advanced Usage
Custom Enhancement Pipeline

from image_enhancer import AdvancedImageEnhancer, EnhancementConfig, EnhancementMode

# Create custom configuration
config = EnhancementConfig(
    target_size=(3840, 2160),  # 4K
    mode=EnhancementMode.VIBRANT,
    denoise_strength=3.5,
    sharpening_strength=1.4,
    contrast_boost=1.3,
    saturation_boost=1.2
)

# Create enhancer and process
enhancer = AdvancedImageEnhancer(config)
result = enhancer.enhance_image("input.jpg", "output.jpg")

print(f"Sharpness improvement: {result.quality_metrics.sharpness:+.1f}")
print(f"Processing time: {result.processing_time:.2f}s")


Integration with Web Applications

from fastapi import FastAPI, File, UploadFile
from image_enhancer import AdvancedImageEnhancer
import cv2
import numpy as np

app = FastAPI()
enhancer = AdvancedImageEnhancer()

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Enhance
    enhanced = enhancer.enhance_image_buffer(image)
    
    # Return enhanced image
    _, buffer = cv2.imencode('.jpg', enhanced)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

    
Shell Script Automation

# enhance_all.sh - Process all images in directory

INPUT_DIR="$1"
OUTPUT_DIR="${INPUT_DIR}_enhanced"

echo "Enhancing images in: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

python image_enhancer.py \
  --input-dir "$INPUT_DIR" \
  --batch \
  --mode natural \
  --size 1080p \
  --denoise 3.0 \
  --sharpen 1.2 \
  --verbose

echo "Enhancement complete!"
echo "See enhancement_report.json for details"


📊 Quality Metrics Explained
The enhancer measures several quality metrics:

Metric	What It Measures	Ideal Range
Sharpness	Edge definition	Higher is better
Contrast	Tonal range	Moderate increase
SNR	Signal-to-Noise Ratio	Higher is better
Brightness	Overall luminance	Minimal change
Color Variance	Color intensity	Slight increase for vibrancy


📁 Project Structure

advanced-image-enhancer/
├── image_enhancer.py     # Main enhancement engine
├── requirements.txt      # Dependencies
├── README.md            # This file
├── examples/            # Example images and results
│   ├── before/          # Original images
│   ├── after/           # Enhanced images
│   └── comparisons/     # Before/after comparisons
├── tests/               # Test suite
│   ├── test_metrics.py  # Quality metric tests
│   └── test_pipeline.py # Pipeline tests
└── docs/                # Documentation
    ├── algorithms.md    # Technical details
    └── api.md          # API documentation

    
⚙️ Processing Pipeline

Image Analysis - Analyze quality metrics
Intelligent Resizing - Preserve quality during resizing
Non-Local Means Denoising - Reduce noise while keeping details
CLAHE Contrast Enhancement - Local contrast improvement
Adaptive Sharpening - Edge-aware sharpening
Color Enhancement - Saturation and vibrancy boost
Quality Validation - Ensure no artifacts introduced


🤝 Contributing
We welcome contributions from developers, photographers, and image processing enthusiasts!


Development Setup

# Clone and setup
git clone https://github.com/yourusername/advanced-image-enhancer.git
cd advanced-image-enhancer

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black image_enhancer.py


Areas for Contribution

New Algorithms - Implement cutting-edge enhancement techniques
GPU Acceleration - Add CUDA support for faster processing
Web Interface - Create a user-friendly web application
Plugin System - Support for custom enhancement filters
More Formats - Support for RAW, HEIC, WebP formats
AI Integration - Incorporate machine learning models


📝 License
This project is licensed under the MIT License - see the LICENSE file for details.


🙏 Acknowledgments

OpenCV for computer vision algorithms
Research in image quality assessment and enhancement
The photography community for feedback and testing
All contributors who help improve this tool


📚 References

"Non-Local Means Denoising" - Buades, Coll, Morel (2005)
"Contrast Limited Adaptive Histogram Equalization" - Zuiderveld (1994)
"Image Quality Assessment" - Wang, Bovik (2006)
"Adaptive Image Sharpening" - Polesel, Ramponi, Mathews (1997)


⚠️ Disclaimer
Always keep backups of original images. While the enhancer is designed to be safe, unexpected results can occur with certain image types. Test on copies before processing important files.


Made with ❤️ for beautiful images everywhere
⭐ If this tool improves your photos, please consider starring the repository!
