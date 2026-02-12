"""
Image Processor Module - RealESRGAN GPU-accelerated image enhancement

Generates 14 output images (7 sizes x 2 formats: JPG + WebP) from a single input image.
Uses InsightFace for face-aware cropping and RealESRGAN x4 for smart upscaling.

Output sizes:
    75x75, 150x150, 250x250, 300x300, 323x323, 1080x1080, 1080x1780

GPU: Uses NVIDIA GPU via CUDA for RealESRGAN inference.
"""

import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# RealESRGAN imports
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    ESRGAN_AVAILABLE = False
    print("[ImageProcessor] WARNING: RealESRGAN not installed. Install with: pip install realesrgan basicsr")

# InsightFace for face detection
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[ImageProcessor] WARNING: InsightFace not installed")

# Check GPU availability
import torch
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"[ImageProcessor] GPU detected: {GPU_NAME}")
else:
    print("[ImageProcessor] WARNING: No GPU detected, RealESRGAN will use CPU (slower)")

# Output size configurations
OUTPUT_SIZES = [
    (75, 75),
    (150, 150),
    (250, 250),
    (300, 300),
    (323, 323),
    (1080, 1080),
    (1080, 1780),
]

# Output formats
OUTPUT_FORMATS = ["jpg", "webp"]

# Quality settings
JPG_QUALITY = 95
WEBP_QUALITY = 95

# Singleton instances
_esrgan_model = None
_face_analyzer = None


def _get_esrgan_model():
    """Initialize RealESRGAN x4 model (singleton, GPU-accelerated)."""
    global _esrgan_model
    if _esrgan_model is not None:
        return _esrgan_model

    if not ESRGAN_AVAILABLE:
        print("[ImageProcessor] RealESRGAN not available")
        return None

    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "weights",
            "RealESRGAN_x4plus.pth"
        )

        # If model not found locally, let realesrgan download it
        if not os.path.exists(model_path):
            model_path = None  # Will auto-download

        _esrgan_model = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,  # 0 for no tile, increase if GPU OOM
            tile_pad=10,
            pre_pad=0,
            half=GPU_AVAILABLE,  # FP16 on GPU for speed
            gpu_id=0 if GPU_AVAILABLE else None,
        )
        device = "GPU" if GPU_AVAILABLE else "CPU"
        print(f"[ImageProcessor] RealESRGAN x4 model loaded on {device}")
        return _esrgan_model

    except Exception as e:
        print(f"[ImageProcessor] Failed to load RealESRGAN model: {e}")
        return None


def _get_face_analyzer():
    """Initialize InsightFace analyzer (singleton)."""
    global _face_analyzer
    if _face_analyzer is not None:
        return _face_analyzer

    if not INSIGHTFACE_AVAILABLE:
        print("[ImageProcessor] InsightFace not available")
        return None

    try:
        _face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        _face_analyzer.prepare(ctx_id=0 if GPU_AVAILABLE else -1, det_size=(640, 640))
        print("[ImageProcessor] InsightFace face analyzer initialized")
        return _face_analyzer

    except Exception as e:
        print(f"[ImageProcessor] Failed to init InsightFace: {e}")
        return None


def detect_face_bbox(image_cv2):
    """
    Detect the primary face bounding box using InsightFace.

    Returns (x1, y1, x2, y2) of the largest face, or None if no face found.
    """
    analyzer = _get_face_analyzer()
    if analyzer is None:
        return None

    try:
        faces = analyzer.get(image_cv2)
        if not faces:
            return None

        # Pick the largest face by bounding box area
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = largest_face.bbox.astype(int)
        return tuple(bbox)  # (x1, y1, x2, y2)

    except Exception as e:
        print(f"[ImageProcessor] Face detection error: {e}")
        return None


def smart_crop(image_cv2, target_w, target_h, face_bbox=None):
    """
    Smart face-centric crop.

    If face_bbox is provided, centers the crop on the face.
    Otherwise, center-crops the image.
    """
    h, w = image_cv2.shape[:2]
    target_ratio = target_w / target_h

    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = face_bbox
        face_cx = (fx1 + fx2) // 2
        face_cy = (fy1 + fy2) // 2
    else:
        face_cx = w // 2
        face_cy = h // 2

    # Determine crop dimensions maintaining target aspect ratio
    img_ratio = w / h
    if img_ratio > target_ratio:
        # Image is wider than target - crop width
        crop_h = h
        crop_w = int(h * target_ratio)
    else:
        # Image is taller than target - crop height
        crop_w = w
        crop_h = int(w / target_ratio)

    # Center crop on face
    x1 = max(0, face_cx - crop_w // 2)
    y1 = max(0, face_cy - crop_h // 2)

    # Ensure crop doesn't go out of bounds
    if x1 + crop_w > w:
        x1 = w - crop_w
    if y1 + crop_h > h:
        y1 = h - crop_h

    x1 = max(0, x1)
    y1 = max(0, y1)

    cropped = image_cv2[y1:y1 + crop_h, x1:x1 + crop_w]
    return cropped


def enhance_with_esrgan(image_cv2):
    """
    Enhance/upscale image using RealESRGAN x4 (GPU).

    Returns the enhanced image or the original if ESRGAN is not available.
    """
    model = _get_esrgan_model()
    if model is None:
        print("[ImageProcessor] ESRGAN not available, skipping enhancement")
        return image_cv2

    try:
        output, _ = model.enhance(image_cv2, outscale=4)
        return output
    except Exception as e:
        print(f"[ImageProcessor] ESRGAN enhancement failed: {e}")
        # Try with tiling if OOM
        try:
            model.tile = 400
            output, _ = model.enhance(image_cv2, outscale=4)
            model.tile = 0  # Reset
            return output
        except Exception as e2:
            print(f"[ImageProcessor] ESRGAN tiled enhancement also failed: {e2}")
            return image_cv2


def process_image_for_sizes(image_path: str, output_dir: str = None, base_name: str = None):
    """
    Process a single image to generate 14 output images (7 sizes x 2 formats).

    Steps:
    1. Detect face in the image
    2. For each target size:
       a. Smart crop around the face with the target aspect ratio
       b. If the crop is smaller than target, enhance with RealESRGAN x4
       c. Resize to exact target dimensions
       d. Save as both JPG and WebP

    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images (default: temp dir)
        base_name: Base name for output files (default: derived from input filename)

    Returns:
        dict with:
            - success: bool
            - output_dir: str - directory containing output images
            - images: list of dicts with {path, size, format, width, height, file_size_kb}
            - processing_time: float
            - gpu_used: bool
            - error: str or None
    """
    start_time = time.time()
    result = {
        "success": False,
        "output_dir": None,
        "images": [],
        "processing_time": 0,
        "gpu_used": GPU_AVAILABLE,
        "error": None,
    }

    try:
        # Read image
        image_cv2 = cv2.imread(image_path)
        if image_cv2 is None:
            # Try with PIL for formats like WebP
            pil_img = Image.open(image_path)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            image_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if image_cv2 is None:
            result["error"] = f"Failed to read image: {image_path}"
            return result

        h, w = image_cv2.shape[:2]
        print(f"[ImageProcessor] Input image: {w}x{h}")

        # Setup output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="photo_processed_")
        os.makedirs(output_dir, exist_ok=True)
        result["output_dir"] = output_dir

        # Base name for output files
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Detect face for smart cropping
        face_bbox = detect_face_bbox(image_cv2)
        if face_bbox:
            print(f"[ImageProcessor] Face detected at: {face_bbox}")
        else:
            print("[ImageProcessor] No face detected, using center crop")

        # Process each target size
        for target_w, target_h in OUTPUT_SIZES:
            size_label = f"{target_w}x{target_h}"
            print(f"[ImageProcessor] Processing size: {size_label}")

            # Smart crop with face centering
            cropped = smart_crop(image_cv2, target_w, target_h, face_bbox)
            crop_h, crop_w = cropped.shape[:2]

            # Check if enhancement is needed (crop smaller than target)
            if crop_w < target_w or crop_h < target_h:
                print(f"[ImageProcessor]   Crop {crop_w}x{crop_h} < target {size_label}, enhancing with ESRGAN")
                cropped = enhance_with_esrgan(cropped)
                crop_h, crop_w = cropped.shape[:2]
                print(f"[ImageProcessor]   Enhanced to: {crop_w}x{crop_h}")

            # Resize to exact target dimensions
            resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            # Save in both formats
            for fmt in OUTPUT_FORMATS:
                filename = f"{base_name}_{target_w}x{target_h}.{fmt}"
                filepath = os.path.join(output_dir, filename)

                if fmt == "jpg":
                    cv2.imwrite(filepath, resized, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
                elif fmt == "webp":
                    cv2.imwrite(filepath, resized, [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])

                file_size_kb = os.path.getsize(filepath) / 1024

                result["images"].append({
                    "path": filepath,
                    "filename": filename,
                    "format": fmt,
                    "width": target_w,
                    "height": target_h,
                    "size_label": size_label,
                    "file_size_kb": round(file_size_kb, 1),
                })

        result["success"] = True
        result["processing_time"] = round(time.time() - start_time, 3)
        print(f"[ImageProcessor] Generated {len(result['images'])} images in {result['processing_time']}s")

    except Exception as e:
        result["error"] = str(e)
        result["processing_time"] = round(time.time() - start_time, 3)
        print(f"[ImageProcessor] Error processing image: {e}")

    return result


def cleanup_processed_images(output_dir: str):
    """Remove the temporary processed images directory."""
    try:
        if output_dir and os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            print(f"[ImageProcessor] Cleaned up: {output_dir}")
    except Exception as e:
        print(f"[ImageProcessor] Cleanup warning: {e}")


# Quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"\nProcessing: {image_path}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"ESRGAN Available: {ESRGAN_AVAILABLE}")
    print(f"InsightFace Available: {INSIGHTFACE_AVAILABLE}")
    print()

    result = process_image_for_sizes(image_path)

    if result["success"]:
        print(f"\nSUCCESS - {len(result['images'])} images generated")
        print(f"Output directory: {result['output_dir']}")
        print(f"Processing time: {result['processing_time']}s")
        print(f"GPU used: {result['gpu_used']}")
        print()
        for img in result["images"]:
            print(f"  {img['filename']} - {img['file_size_kb']} KB")
    else:
        print(f"\nFAILED: {result['error']}")
