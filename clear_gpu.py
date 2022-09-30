#!/opt/conda/bin/python
import torch
import os

print(f"Clearing CUDA_ID:{os.environ.get('CUDA_ID')}")
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(7, True))