import torch
from torch.optim import AdamW

class TopRAdamW(AdamW):
    def __init__(self, params, lr=1e-3, top_r=1.0, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        assert 0 < top_r <= 1.0
        self.top_r = top_r

    @torch.no_grad()
    def step(self, closure=None):
        if self.top_r < 1.0:  
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    g = p.grad
                    flat = g.view(-1)
                    numel = flat.numel()
                    if numel == 0:
                        continue

                    k = int(self.top_r * numel)

                    if k <= 0:
                        g.zero_()
                        continue

                    if k >= numel:
                        continue

                    thresh = flat.abs().kthvalue(numel - k).values

                    mask = (flat.abs() >= thresh)
                    flat.mul_(mask)

        return super().step(closure)
