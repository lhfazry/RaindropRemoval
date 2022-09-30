#!/opt/conda/bin/python
import torch
import os

torch.cuda.init()


print(f"Clearing CUDA_ID:{os.environ.get('CUDA_ID')}")
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(int(os.environ.get('CUDA_ID')), True))