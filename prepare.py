from pathlib import Path
import gdown
import zipfile
import os

Path("logs/LEVIR").mkdir(parents=True, exist_ok=True)
Path("datasets/LEVIR").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)


if len(os.listdir("datasets/LEVIR")) == 0:
    # download datasets
    gdown.download(id="1-0r113orm-ETks8pYG-qUSTZQPry4L4m", output="datasets/LEVIR.zip", quiet=False)

    #extract datasets
    with zipfile.ZipFile('datasets/LEVIR.zip', 'r') as zip_ref:
        zip_ref.extractall('datasets/LEVIR')
        os.remove('datasets/LEVIR.zip')

