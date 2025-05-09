# Capstone Country 🌐🏙️  
**Multi-source Street-view Country Detection (PyTorch)**

This repository is a refactored fork of **[YoungIT/godeye-core]** that

* ensembles **StreetClip** (CLIP-ViT zero-shot country head) with  
  **TIB Hannover GeoEstimator** (ResNet-50 coordinate regressor)
* optionally fuses a lightweight **traffic-sign CNN** as a 2ᵈ modality
* adds a tiny **real-time “drag-a-box-on-screen → country”** helper (`realtime.py`)

Tested on Windows 10/11 with an RTX 4080 GPU.

---

## Quick start

```bash
# clone THIS repo (or pull if you're already inside it)
git clone --recursive https://github.com/hoon9732/capstone_country.git
cd capstone_country

# 1. create env  (Python 3.10 works best)
conda create -n geoenv python=3.10 -y
conda activate geoenv
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install pyautogui opencv-python

# 2. grab the pre-trained ResNet checkpoint (18 MB)
mkdir -p resources/tibhannover/models
curl -L -o resources/tibhannover/models/epoch=014-val_loss=18.4833.ckpt ^
  https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/epoch.014-val_loss.18.4833.ckpt

# 3. sanity check (CLI)
python src/core/core.py img=assets/imgs/london.jpeg \
       geo_estimation=tibhannover          # <— note the underscore
