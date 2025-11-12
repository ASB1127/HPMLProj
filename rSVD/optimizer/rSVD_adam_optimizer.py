import math
import torch
from torch.optim.optimizer import Optimizer


def _flatten_last_dims(t):
    """
    View a tensor as matrix (m, n) where m = t.shape[0], n = prod(t.shape[1:]).
    Returns (mat, unflatten_fn) where unflatten_fn maps (m, n) back to t.shape.
    """
    orig_shape = t.shape
    if t.ndim == 1:
        # Treat as (m, n) = (1, N)
        mat = t.view(1, -1)
        def unflatten(x):  # x is (1, N)
            return x.view(orig_shape)
        return mat, unflatten
    else:
        m = t.shape[0]
        n = int(torch.prod(torch.tensor(t.shape[1:], device=t.device)))
        mat = t.view(m, n)
        def unflatten(x):  # x is (m, n)
            return x.view(orig_shape)
        return mat, unflatten


class rSVDAdam(Optimizer):
    r"""
    Adam / AdamW with Randomized SVD (rSVD).

    For matrix-like params (ndim >= 2), gradients are projected to a low-rank subspace
    using randomized SVD every `proj_interval` steps. Adam moments are tracked in the
    reduced space (r x n), then reconstructed back to full space for the update.

    For 1-D params (e.g., bias), falls back to vanilla Adam.

    Args:
        params: iterable of tensors to optimize.
        lr: learning rate (default: 1e-3).
        betas: Adam betas (default: (0.9, 0.999)).
        eps: numerical stability epsilon (default: 1e-8).
        weight_decay: weight decay strength (default: 0.0).
        amsgrad: AMSGrad variant (default: False).
        decoupled_weight_decay: if True, AdamW-style decay; else classic L2.
        use_rgp: enable rSVD for eligible tensors (default: True).
        rank_fraction: fraction of min(m, n) to use as target rank r (default: 0.25).
        proj_interval: recompute projector P every N steps (default: 200).
        oversample: randomized SVD oversampling (k = r + oversample) (default: 4).
        power_iters: randomized SVD power iterations (default: 1).
        verbose_memory_once: print estimated state memory once per tensor (default: True).
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
        *,
        use_rgp: bool = True,
        rank_fraction: float = 0.25,
        proj_interval: int = 200,
        oversample: int = 4,
        power_iters: int = 1,
        verbose_memory_once: bool = True,
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
        if not (0.0 < rank_fraction <= 1.0):
            raise ValueError(f"rank_fraction must be in (0,1], got {rank_fraction}")
        if proj_interval < 1:
            raise ValueError(f"proj_interval must be >= 1, got {proj_interval}")
        if oversample < 0:
            raise ValueError(f"oversample must be >= 0, got {oversample}")
        if power_iters < 0:
            raise ValueError(f"power_iters must be >= 0, got {power_iters}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            decoupled_weight_decay=decoupled_weight_decay,
            use_rgp=use_rgp,
            rank_fraction=rank_fraction,
            proj_interval=proj_interval,
            oversample=oversample,
            power_iters=power_iters,
            verbose_memory_once=verbose_memory_once,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _maybe_print_memory(self, p, state, m, n, r, dtype):
        # Print a one-time estimate of optimizer-state memory (baseline Adam vs rSVD) for this tensor.
        if state.get("_printed_mem_once", False):
            return
        elem_size = torch.tensor([], dtype=dtype).element_size()

        # Baseline Adam states: 2 * (m*n) elements (exp_avg, exp_avg_sq)
        baseline_elems = 2 * m * n
        # rSVD states: P(m*r) + M(r*n) + V(r*n) + (maybe max_exp_avg_sq_low r*n if AMSGrad)
        amsgrad = state.get("_amsgrad_enabled", False)
        rgp_elems = (m * r) + (r * n) + (r * n) + (r * n if amsgrad else 0)

        baseline_mb = baseline_elems * elem_size / (1024 ** 2)
        rgp_mb = rgp_elems * elem_size / (1024 ** 2)

        print(
            f"[rSVDAdam] Param shape={tuple(p.shape)} | "
            f"Adam states ≈ {baseline_mb:.2f} MB vs rSVD states ≈ {rgp_mb:.2f} MB "
            f"(r={r}, save ≈ {baseline_mb - rgp_mb:.2f} MB)"
        )
        state["_printed_mem_once"] = True

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with rSVD for matrix-like params."""
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
            use_rgp = group["use_rgp"]
            rank_fraction = group["rank_fraction"]
            proj_interval = group["proj_interval"]
            oversample = group["oversample"]
            power_iters = group["power_iters"]
            verbose_memory_once = group["verbose_memory_once"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("rSVDAdam does not support sparse gradients; use SparseAdam instead.")

                state = self.state[p]
                state["_amsgrad_enabled"] = amsgrad  # for memory print

                # === Classic (L2) weight decay: add to gradient before moments.
                eff_grad = grad
                if wd != 0.0 and not decoupled:
                    eff_grad = eff_grad.add(p, alpha=wd)

                # === Decide path: vanilla Adam for 1-D or when rSVD disabled ===
                if (not use_rgp) or (eff_grad.ndim < 2):
                    # Initialize vanilla Adam states if missing or mismatched
                    if (len(state) == 0 or
                        state.get("exp_avg") is None or
                        state.get("exp_avg_sq") is None or
                        state["exp_avg"].shape != p.shape or
                        state["exp_avg_sq"].shape != p.shape):
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] = state.get("step", 0) + 1
                    t = state["step"]

                    # Moments
                    exp_avg.mul_(beta1).add_(eff_grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(eff_grad, eff_grad, value=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t

                    # Denominator
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    step_size = lr / bias_correction1

                    # Decoupled weight decay (AdamW)
                    if wd != 0.0 and decoupled:
                        p.mul_(1 - lr * wd)

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    continue  # next parameter

                # === rSVD path for matrix-like params (ndim >= 2) ===
                # Flatten to (m, n)
                G_mat, unflatten = _flatten_last_dims(eff_grad)
                m, n = G_mat.shape
                # Choose rank
                r_max = max(1, min(m, n))
                r = max(1, int(rank_fraction * min(m, n)))
                r = min(r, r_max)

                # Initialize / reinit state for rSVD
                # Stored:
                # - step
                # - proj: P (m x r)
                # - exp_avg_low: (r x n), exp_avg_sq_low: (r x n)
                # - (optional) max_exp_avg_sq_low: (r x n)
                need_init = (
                    (len(state) == 0) or
                    ("exp_avg_low" not in state) or
                    ("exp_avg_sq_low" not in state) or
                    ("proj" not in state)
                )
                shape_mismatch = False
                if not need_init:
                    P = state["proj"]
                    exp_avg_low = state["exp_avg_low"]
                    exp_avg_sq_low = state["exp_avg_sq_low"]
                    shape_mismatch = (
                        P.shape[0] != m or P.shape[1] != r or
                        exp_avg_low.shape != (r, n) or
                        exp_avg_sq_low.shape != (r, n)
                    )

                if need_init or shape_mismatch:
                    state["step"] = 0
                    # Initialize with dummy projector; will compute below.
                    state["proj"] = torch.zeros((m, r), device=p.device, dtype=p.dtype)
                    state["exp_avg_low"] = torch.zeros((r, n), device=p.device, dtype=p.dtype)
                    state["exp_avg_sq_low"] = torch.zeros((r, n), device=p.device, dtype=p.dtype)
                    if amsgrad:
                        state["max_exp_avg_sq_low"] = torch.zeros((r, n), device=p.device, dtype=p.dtype)

                # Step count
                state["step"] = state.get("step", 0) + 1
                t = state["step"]

                # Periodically (re)compute projector P via randomized SVD
                recompute_P = (t % proj_interval == 1) or need_init or shape_mismatch
                if recompute_P:
                    # Using torch.svd_lowrank for efficiency
                    q = min(r + oversample, min(m, n))
                    # Power iterations are supported via 'niter' in svd_lowrank
                    U, S, Vh = torch.svd_lowrank(G_mat, q=q, niter=power_iters)
                    # Take first r left singular vectors
                    P = U[:, :r].contiguous()
                    state["proj"] = P

                else:
                    P = state["proj"]

                # Project gradient to low-rank space: R = P^T G
                R = P.transpose(0, 1).matmul(G_mat)  # (r, n)
                assert R.shape == (r, n), f"Projected grad shape mismatch: {R.shape} vs {(r, n)}"

                # Fetch low-rank states
                exp_avg_low = state["exp_avg_low"]
                exp_avg_sq_low = state["exp_avg_sq_low"]

                # Update moments in low-rank space
                exp_avg_low.mul_(beta1).add_(R, alpha=1 - beta1)
                exp_avg_sq_low.mul_(beta2).addcmul_(R, R, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # Denominator in low-rank space
                if amsgrad:
                    max_exp_avg_sq_low = state.get("max_exp_avg_sq_low", None)
                    if max_exp_avg_sq_low is None or max_exp_avg_sq_low.shape != (r, n):
                        max_exp_avg_sq_low = torch.zeros((r, n), device=p.device, dtype=p.dtype)
                        state["max_exp_avg_sq_low"] = max_exp_avg_sq_low
                    torch.maximum(max_exp_avg_sq_low, exp_avg_sq_low, out=max_exp_avg_sq_low)
                    denom_low = (max_exp_avg_sq_low.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom_low = (exp_avg_sq_low.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                # Decoupled weight decay (AdamW): scale weights directly
                if wd != 0.0 and decoupled:
                    p.mul_(1 - lr * wd)

                # Low-rank normalized update, then lift back: update_full = P @ (exp_avg_low / denom_low)
                update_low = exp_avg_low / denom_low
                update_full = P.matmul(update_low)  # (m, n)
                assert update_full.shape == (m, n), f"Full update shape mismatch: {update_full.shape} vs {(m, n)}"

                # Apply to parameter
                p.add_(unflatten(update_full), alpha=-step_size)

                # Optional one-time memory print
                if verbose_memory_once:
                    self._maybe_print_memory(
                        p, state, m=m, n=n, r=r, dtype=p.dtype
                    )

        return loss
