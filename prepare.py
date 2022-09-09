#!/opt/conda/bin/python
from pathlib import Path
import gdown
import zipfile
import os

#Path("logs/LEVIR").mkdir(parents=True, exist_ok=True)
Path("datasets/RainDrop").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)


if len(os.listdir("datasets/RainDrop")) == 0:
    # download datasets
    gdown.download(id="1--p0N1qSOCCLhnmGtvl-rAAOozq-1d-E", output="datasets/RainDrop/train.zip", quiet=False)
    gdown.download(id="18KA9FHiYM3nvqpSZi28LjyAGyABGsb0D", output="datasets/RainDrop/test_a.zip", quiet=False)
    gdown.download(id="1f1IEHZnmWqWErL87K-RVid3dg7ZiXnk5", output="datasets/RainDrop/test_b.zip", quiet=False)

    for ds in ['test_a', 'test_b', 'train']:
        #extract datasets
        with zipfile.ZipFile(f'datasets/RainDrop/{ds}.zip', 'r') as zip_ref:
            zip_ref.extractall('datasets/RainDrop')
            os.remove(f'datasets/RainDrop/{ds}.zip')


