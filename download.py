#!/opt/conda/bin/python
from pathlib import Path
import gdown
import zipfile
import os

#Path("logs/LEVIR").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

if not os.path.exists('models/LEVIR_ema_0.9999_650000.pt'):
    gdown.download(id="1bcnQE5XiU8CMIzzoykQbgW9oFBX3y1xC", output="models/LEVIR_ema_0.9999_650000.pt", quiet=False)