import torch
from functools import partial

a = torch.randn((20))
b = torch.randn((20))
c = a / b
print(c.max())
print(c.min())

maybe_clip = partial(torch.clamp, min = -1., max = 1.)
d = maybe_clip(c)
print(d.max())
print(d.min())