import torch
import numpy as np

# torch数据转nump
a=torch.ones((2,3),dtype=int)
b=a.numpy()
print(b)
# numpy数据转torch
c=np.arange(0,12,2)
d=torch.from_numpy(c)
print(d)