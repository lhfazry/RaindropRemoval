#!/opt/conda/bin/python
import torch

torch.cuda.empty_cache()
print(torch.cuda.memory_summary())