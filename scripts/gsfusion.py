import cv2 as cv
import numpy as np
from PIL import Image

#code for forward transformation of GS fusion algorithm 
def compute_gs_components(bands, gs1):
    """
    Implements Step 2 of the GS fusion algorithm:
    Forward transformation to get GS2, GS3, ...

    Args:
        bands: list of 2D numpy arrays (original MS bands)
        gs1: 2D numpy array (simulated panchromatic image)

    Returns:
        gs_components: list of GS components (GS1, GS2, ...)
    """
    bands = [band.astype(np.float32) for band in bands]
    gs1 = gs1.astype(np.float32)
    gs_components = [gs1]  # GS1 is already given

    for t, B_T in enumerate(bands):
        B_T_mean = np.mean(B_T)
        B_T_centered = B_T - B_T_mean

        proj_sum = np.zeros_like(B_T)

        for l in range(len(gs_components)):
            GS_l = gs_components[l]
            GS_l_centered = GS_l - np.mean(GS_l)

            # Cov(B_T, GS_l) / Var(GS_l)
            numerator = np.mean(B_T_centered * GS_l_centered)
            denominator = np.mean(GS_l_centered ** 2)
            phi = numerator / denominator

            proj_sum += phi * GS_l_centered

        GS_T = B_T_centered - proj_sum
        gs_components.append(GS_T)

    return gs_components

def match_gray_levels(pan, gs1):
    # Convert to float for safe computation
    pan = pan.astype(np.float32)
    gs1 = gs1.astype(np.float32)

    mu_pan = np.mean(pan)
    sigma_pan = np.std(pan)

    mu_gs1 = np.mean(gs1)
    sigma_gs1 = np.std(gs1)

    # Gray level matching
    pan_matched = mu_gs1 + (sigma_gs1 / sigma_pan) * (pan - mu_pan)
    return pan_matched

#backward tranformation of GS fusion algorithm 
def inverse_gs_transformation(gs_components, original_bands):
    """
    Reconstructs the multispectral bands from the GS components using the inverse GS formula.
    
    Args:
        gs_components: List of GS components after PAN has replaced GS1
        original_bands: List of original MS bands (used to get means and phi coefficients)
        
    Returns:
        fused_bands: List of fused multispectral bands
    """
    fused_bands = []
    
    for t in range(1, len(gs_components)):  # GS1 is replaced PAN; skip index 0
        GS_T = gs_components[t]
        B_T = original_bands[t - 1].astype(np.float32)  # t-1 because band indexing starts from 0

        mu_T = np.mean(B_T)
        proj_sum = np.zeros_like(B_T)

        for l in range(t):  # l from 0 to t-1
            GS_l = gs_components[l]
            GS_l_centered = GS_l - np.mean(GS_l)
            B_T_centered = B_T - mu_T

            # Regression coefficient
            phi = np.mean(B_T_centered * GS_l_centered) / np.mean(GS_l_centered ** 2)

            proj_sum += phi * GS_l_centered

        # Inverse GS formula
        B_T_reconstructed = GS_T + mu_T + proj_sum
        fused_bands.append(B_T_reconstructed)

    return fused_bands

def run_pansharpening(ms_path, pan_path):
    img_MS = cv.imread(ms_path)
    img_PAN = cv.imread(pan_path)

    # Convert MS to R, G, B bands
    b, g, r = cv.split(img_MS)
    red_array = np.array(r)
    green_array = np.array(g)
    blue_array = np.array(b)

    wr, wg, wb = 0.4, 0.4, 0.2
    gs1 = wr * red_array + wg * green_array + wb * blue_array
    bands = [red_array, green_array, blue_array]

    gs_components = compute_gs_components(bands, gs1)

    b_pan, g_pan, r_pan = cv.split(img_PAN)
    pan = b_pan  # assumes blue channel is PAN

    pan_matched = match_gray_levels(pan, gs1)

    # Replace GS1 with matched PAN
    gs_components[0] = pan_matched

    fused = inverse_gs_transformation(gs_components, bands)
    fused_stack = cv.merge([fused[2], fused[1], fused[0]])  # back to BGR
    fused_uint8 = np.clip(fused_stack, 0, 255).astype(np.uint8)

    # Convert to RGB for PIL
    fused_rgb = cv.cvtColor(fused_uint8, cv.COLOR_BGR2RGB)
    fused_img_pil = Image.fromarray(fused_rgb)

    return fused_img_pil