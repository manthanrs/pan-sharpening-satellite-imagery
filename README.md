# 🛰️ Pan-Sharpening of Satellite Imagery using GS and IHS Fusion

This project implements and compares two popular pan-sharpening techniques:
- Gram-Schmidt (GS) Fusion
- Intensity-Hue-Saturation (IHS) Fusion

## 📂 Structure
- `scripts/`: Contains modular Python scripts for each fusion method
- `data/`: Input data 
- `outputs/`: Visual results from each fusion technique

## 🛠️ Tech Stack
- Python
- OpenCV
- NumPy
- Streamlit

## 📊 Results
| Fusion Method | Visual Quality | Notes |
|---------------|----------------|-------|
| GS Fusion     | ✅ Natural colors retained | Best spectral fidelity |
| IHS Fusion    | ⚠️ Slight blue tint (color distortion) | Faster & simpler |

## 🚀 Getting Started
```bash
git clone https://github.com/yourusername/pan-sharpening-satellite-imagery.git
cd pan-sharpening-satellite-imagery
pip install -r requirements.txt
