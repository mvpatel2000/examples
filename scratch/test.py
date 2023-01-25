import torch
from composer.utils import dist

dist.initialize_dist('gpu')

a = torch.tensor(1, device='cuda')
dist.all_reduce(a)
print(a)