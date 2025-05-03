import numpy as np
import cv2 as cv
import math
from PIL import Image

def match_histogram(pan, intensity):
    pan = pan.astype(np.float32)
    intensity = intensity.astype(np.float32)

    mu_pan, sigma_pan = np.mean(pan), np.std(pan)
    mu_I, sigma_I = np.mean(intensity), np.std(intensity)

    pan_matched = mu_I + (sigma_I / sigma_pan) * (pan - mu_pan)
    return np.clip(pan_matched, 0, 255).astype(np.uint8)

def ihs_to_bgr(hsi_img):
    rows, cols = hsi_img.shape[:2]
    bgr = np.zeros_like(hsi_img, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            H = hsi_img[i, j, 0] * math.pi / 180  # Convert degrees to radians
            S = hsi_img[i, j, 1] / 255.0
            I = hsi_img[i, j, 2] / 255.0

            if S == 0:
                r = g = b = I
            else:
                if 0 <= H < 2 * math.pi / 3:
                    B = I * (1 - S)
                    R = I * (1 + (S * math.cos(H)) / math.cos(math.pi / 3 - H))
                    G = 3 * I - (R + B)
                elif 2 * math.pi / 3 <= H < 4 * math.pi / 3:
                    H -= 2 * math.pi / 3
                    R = I * (1 - S)
                    G = I * (1 + (S * math.cos(H)) / math.cos(math.pi / 3 - H))
                    B = 3 * I - (R + G)
                else:
                    H -= 4 * math.pi / 3
                    G = I * (1 - S)
                    B = I * (1 + (S * math.cos(H)) / math.cos(math.pi / 3 - H))
                    R = 3 * I - (G + B)

                r, g, b = R, G, B

            bgr[i, j] = [b * 255, g * 255, r * 255]

    return np.clip(bgr, 0, 255).astype(np.uint8)

def run_ihs_technique(ms_path, pan_path):
    img_MS = cv.imread(ms_path)
    img_PAN = cv.imread(pan_path)

    rows, cols = img_MS.shape[:2]
    hsi = np.zeros_like(img_MS, dtype=np.float32)

    # RGB to HSI conversion
    for i in range(rows):
        for j in range(cols):
            b, g, r = img_MS[i, j] / 255.0  
            I = (r + g + b) / 3.0
            min_val = min(r, g, b)
            sum_rgb = r + g + b
            S = 0 if sum_rgb == 0 else 1 - (3 * min_val / sum_rgb)

            if S == 0:
                H = 0
            else:
                num = 0.5 * ((r - g) + (r - b))
                den = math.sqrt((r - g)**2 + (r - b)*(g - b))
                theta = 0 if den == 0 else math.acos(np.clip(num / den, -1, 1))
                H = theta if b <= g else 2 * math.pi - theta

            hsi[i, j, 0] = H * 180 / math.pi
            hsi[i, j, 1] = S * 255
            hsi[i, j, 2] = I * 255

    hsi = np.clip(hsi, 0, 255).astype(np.uint8)

    b, g, r = cv.split(img_PAN)
    pan_matched = match_histogram(b, hsi[:, :, 2])
    hsi[:, :, 2] = pan_matched

    rgb_fused = ihs_to_bgr(hsi)
    fused_rgb = cv.cvtColor(rgb_fused, cv.COLOR_BGR2RGB)
    fused_img_pil = Image.fromarray(fused_rgb)

    return fused_img_pil
