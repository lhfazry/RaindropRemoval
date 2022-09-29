#!/opt/conda/bin/python
from pathlib import Path
import gdown
import zipfile
import os

#Path("logs/LEVIR").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

if not os.path.exists('models/RainDrop_ema_0.9999_1000000.pt'):
    gdown.download(id="1wWMePiseUNCZAD-GuyylwrAmyUtvnPUb", output="models/RainDrop_ema_0.9999_1000000.pt", quiet=False)