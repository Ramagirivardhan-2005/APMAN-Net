"""
Underwater Image Quality Metrics Module
Provides Full-Reference (PSNR, SSIM, MSE) and Non-Reference (UCIQE, UIQM) evaluation.
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def calculate_psnr(img1_np, img2_np):
    """Compute PSNR between two uint8 RGB images [0, 255]."""
    try:
        return float(compute_psnr(img1_np, img2_np))
    except Exception:
        mse = np.mean((img1_np.astype(np.float64) - img2_np.astype(np.float64)) ** 2)
        if mse == 0:
            return 100.0
        return float(10 * np.log10(255.0 ** 2 / mse))


def calculate_ssim(img1_np, img2_np):
    """Compute SSIM between two uint8 RGB images [0, 255]."""
    try:
        return float(compute_ssim(img1_np, img2_np, channel_axis=2))
    except Exception:
        return 0.0


def calculate_mse(img1_np, img2_np):
    """Compute Mean Squared Error."""
    return float(np.mean((img1_np.astype(np.float64) - img2_np.astype(np.float64)) ** 2))


def calculate_uciqe(img_rgb):
    """
    Underwater Color Image Quality Evaluation (UCIQE) metric.
    Yang, M., & Xu, A. (2015). IEEE TIP.
    UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s
    Higher value indicates higher perceptual underwater quality.
    """
    try:
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
        l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
        
        # Chroma standard deviation
        chroma = np.sqrt(a**2 + b**2)
        sigma_c = np.std(chroma)
        
        # Luminance contrast (difference between top 1% and bottom 1%)
        con_l = np.percentile(l, 99) - np.percentile(l, 1)
        
        # Average saturation
        sat = chroma / np.sqrt(chroma**2 + l**2 + 1e-6)
        mu_s = np.mean(sat)
        
        # Standard coefficients
        uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
        return round(float(uciqe), 4)
    except Exception:
        return 0.0


def calculate_uiqm(img_rgb):
    """
    Underwater Image Quality Measure (UIQM).
    Panetta et al. (2015). IEEE JOE.
    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
    Combines:
    - UICM: Underwater Image Colorfulness Measure
    - UISM: Underwater Image Sharpness Measure
    - UIConM: Underwater Image Contrast Measure
    """
    try:
        r = img_rgb[:, :, 0].astype(np.float64)
        g = img_rgb[:, :, 1].astype(np.float64)
        b = img_rgb[:, :, 2].astype(np.float64)
        
        # 1. Colorfulness (UICM)
        rg = r - g
        yb = 0.5 * (r + g) - b
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        var_rg, var_yb = np.var(rg), np.var(yb)
        uicm = -0.0268 * np.sqrt(mean_rg**2 + mean_yb**2) + 0.1586 * np.sqrt(var_rg + var_yb)
        
        # 2. Sharpness (UISM) via Sobel edge magnitude
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(gx**2 + gy**2)
        uism = np.mean(edge_mag) / 10.0
        
        # 3. Contrast (UIConM)
        mean_gray = np.mean(gray) + 1e-6
        std_gray = np.std(gray)
        uiconm = (std_gray / mean_gray) * 2.0
        
        # UIQM composite
        uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
        return round(float(uiqm), 4)
    except Exception:
        return 0.0


def compute_all_metrics(raw_np, enhanced_np, ref_np=None):
    """
    Calculates full comparative metric suite between raw and enhanced,
    plus full-reference metrics against ground truth if provided.
    """
    metrics = {
        "raw_uciqe": calculate_uciqe(raw_np),
        "enhanced_uciqe": calculate_uciqe(enhanced_np),
        "raw_uiqm": calculate_uiqm(raw_np),
        "enhanced_uiqm": calculate_uiqm(enhanced_np),
    }
    metrics["uciqe_gain"] = round(metrics["enhanced_uciqe"] - metrics["raw_uciqe"], 4)
    metrics["uiqm_gain"] = round(metrics["enhanced_uiqm"] - metrics["raw_uiqm"], 4)
    
    if ref_np is not None:
        # Resize reference to match enhanced if shapes differ
        if ref_np.shape != enhanced_np.shape:
            ref_np = cv2.resize(ref_np, (enhanced_np.shape[1], enhanced_np.shape[0]))
        if ref_np.shape != raw_np.shape:
            raw_np = cv2.resize(raw_np, (enhanced_np.shape[1], enhanced_np.shape[0]))
            
        metrics["raw_psnr"] = round(calculate_psnr(ref_np, raw_np), 2)
        metrics["enhanced_psnr"] = round(calculate_psnr(ref_np, enhanced_np), 2)
        metrics["psnr_gain"] = round(metrics["enhanced_psnr"] - metrics["raw_psnr"], 2)
        
        metrics["raw_ssim"] = round(calculate_ssim(ref_np, raw_np), 4)
        metrics["enhanced_ssim"] = round(calculate_ssim(ref_np, enhanced_np), 4)
        metrics["ssim_gain"] = round(metrics["enhanced_ssim"] - metrics["raw_ssim"], 4)
        
        metrics["raw_mse"] = round(calculate_mse(ref_np, raw_np), 2)
        metrics["enhanced_mse"] = round(calculate_mse(ref_np, enhanced_np), 2)
    
    return metrics
