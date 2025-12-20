import torch
from torch.optim import AdamW

class TopRAdamW(AdamW):
    """
    AdamW optimizer with Top-R gradient masking.
    Only the top-R fraction of the gradient magnitudes are kept for each parameter.
    """
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
                    if g.is_sparse:
                        continue

                    flat = g.reshape(-1)
                    numel = flat.numel()
                    if numel == 0:
                        continue

                    k = int(self.top_r * numel)

                    if k <= 0:
                        g.zero_()
                        continue

                    if k >= numel:
                        continue

                    # Keep exactly the top-k magnitudes (avoids off-by-one and tie issues).
                    topk_idx = flat.abs().topk(k, largest=True, sorted=False).indices
                    mask = torch.zeros_like(flat, dtype=torch.bool)
                    mask.scatter_(0, topk_idx, True)
                    flat.mul_(mask)

        return super().step(closure)
