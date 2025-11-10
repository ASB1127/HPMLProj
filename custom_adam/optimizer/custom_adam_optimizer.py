import math
import torch
from torch.optim.optimizer import Optimizer


class CustomAdam(Optimizer):
    r"""
    Adam / AdamW (decoupled) optimizer implemented from scratch.

    Args:
        params (iterable): model parameters to optimize.
        lr (float): learning rate (default: 1e-3).
        betas (Tuple[float, float]): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): term added to denominator for numerical stability (default: 1e-8).
        weight_decay (float): weight decay strength (default: 0.0).
        amsgrad (bool): use the AMSGrad variant (default: False).
        decoupled_weight_decay (bool): if True, use AdamW-style decoupled decay.
            If False and weight_decay>0, uses classic L2 (adds to grad).
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        decoupled_weight_decay: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, decoupled_weight_decay=decoupled_weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]
            decoupled = group["decoupled_weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("CustomAdam does not support sparse gradients; use SparseAdam instead.")

                # Classic (L2) weight decay: add to gradient before moments.
                if wd != 0.0 and not decoupled:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]

                # State initialization
                if (len(state) == 0  or state["exp_avg"].shape != p.shape or state["exp_avg_sq"].shape != p.shape):
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                # Decoupled weight decay (AdamW): scale weights directly
                if wd != 0.0 and decoupled:
                    p.mul_(1 - lr * wd)

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

