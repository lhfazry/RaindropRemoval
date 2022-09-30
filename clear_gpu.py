#!/opt/conda/bin/python
import torch

torch.cuda.empty_cache()
torch.cuda.memory_summary()