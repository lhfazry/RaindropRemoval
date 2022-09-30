#!/opt/conda/bin/python
import torch
import os
import gc

gc.collect()
print(f"Clearing CUDA_ID:{os.environ.get('CUDA_ID')}")
torch.cuda.empty_cache()

torch.cuda.init()
print(torch.cuda.memory_summary(int(os.environ.get('CUDA_ID')), True))