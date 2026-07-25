from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from image_enhancer.config import EnhancementConfig


# ── EXIF Orientation ─────────────────────────────────────────────────────

def correct_exif_orientation(image: np.ndarray, image_path: Path) -> np.ndarray:
    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as pil_img:
            exif = pil_img._getexif()
            if exif is not None:
                orientation = exif.get(0x0112, 1)
                if orientation == 3:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif orientation == 6:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == 8:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return image


# ── Load / Save ──────────────────────────────────────────────────────────

def load_image(image_path: Path) -> Optional[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = correct_exif_orientation(image, image_path)
    return image


def save_image(image: np.ndarray, output_path: Path) -> bool:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ext = output_path.suffix.lower()
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, 95]
        cv2.imwrite(str(output_path), image_bgr, params)
        return True
    except Exception:
        return False


# ── Resize ───────────────────────────────────────────────────────────────

def resize_to_target(
    image: np.ndarray,
    target_size: Tuple[int, int],
    fit_mode: str = "stretch",
) -> np.ndarray:
    height, width = image.shape[:2]
    target_width, target_height = target_size

    if (width, height) == (target_width, target_height):
        return image

    if fit_mode == "stretch":
        interp = cv2.INTER_LANCZOS4 if (width < target_width or height < target_height) else cv2.INTER_AREA
        return cv2.resize(image, (target_width, target_height), interpolation=interp)

    if fit_mode == "crop":
        scale = max(target_width / width, target_height / height)
        new_w, new_h = int(width * scale), int(height * scale)
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        start_x = (new_w - target_width) // 2
        start_y = (new_h - target_height) // 2
        return resized[start_y:start_y + target_height, start_x:start_x + target_width]

    if fit_mode == "pad":
        scale = min(target_width / width, target_height / height)
        new_w, new_h = int(width * scale), int(height * scale)
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        start_x = (target_width - new_w) // 2
        start_y = (target_height - new_h) // 2
        result[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        return result

    return image


# ── Denoising ────────────────────────────────────────────────────────────

def apply_non_local_means_denoising(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    strength = float(params["denoise_strength"])
    if strength == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_denoised = cv2.fastNlMeansDenoising(l, None, h=strength * 10, templateWindowSize=7, searchWindowSize=21)
    lab_denoised = cv2.merge([l_denoised, a, b])
    return cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2RGB)


# ── Contrast ─────────────────────────────────────────────────────────────

def enhance_contrast_local(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(params["contrast_boost"]) * 2, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    l_enhanced = cv2.bilateralFilter(l_enhanced, 5, 75, 75)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


# ── Saturation ───────────────────────────────────────────────────────────

def enhance_saturation(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    saturation_factor = float(params["saturation_boost"])
    if saturation_factor == 1.0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
    hsv_enhanced = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)


# ── Sharpening ───────────────────────────────────────────────────────────

def adaptive_sharpening(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    strength = float(params["sharpening_strength"])
    if strength == 1.0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.dilate(edges, None, iterations=1)
    edge_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (5, 5), 1.0)
    edge_mask = np.clip(edge_mask / 255.0, 0, 1)
    if image.ndim == 3:
        edge_mask = np.stack([edge_mask] * 3, axis=2)
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    result = image * (1.0 - edge_mask) + sharpened * edge_mask
    return np.clip(result, 0, 255).astype(np.uint8)


# ── White Balance ────────────────────────────────────────────────────────

def auto_white_balance(image: np.ndarray) -> np.ndarray:
    avg_r = float(np.mean(image[:, :, 0]))
    avg_g = float(np.mean(image[:, :, 1]))
    avg_b = float(np.mean(image[:, :, 2]))
    target = (avg_r + avg_g + avg_b) / 3.0
    if min(avg_r, avg_g, avg_b) == 0:
        return image
    scales = np.array([target / avg_r, target / avg_g, target / avg_b])
    balanced = image.astype(np.float32) * scales.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


# ── Low Light ────────────────────────────────────────────────────────────

def enhance_low_light(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(np.mean(gray))
    gamma = max(0.3, min(2.0, 128.0 / max(mean_brightness, 1.0)))
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


# ── Super Resolution ─────────────────────────────────────────────────────

def apply_super_resolution_effect(image: np.ndarray, config: EnhancementConfig) -> np.ndarray:
    h, w = image.shape[:2]
    upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    params = config.get_mode_params()
    upscaled = enhance_contrast_local(upscaled, params)
    upscaled = enhance_saturation(upscaled, params)
    downscaled = cv2.resize(upscaled, config.target_size, interpolation=cv2.INTER_AREA)
    return downscaled
