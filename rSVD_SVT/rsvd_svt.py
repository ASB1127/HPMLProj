import torch
import transformers

class RandomizedSVDGradientProjector:
    def __init__(self, rank: int = 50, update_freq: int = 100):
        """
        Args:
            rank: Target rank k for SVD approximation
            update_freq: How often (in steps) to recompute SVD
        """
        self.rank = rank
        self.update_freq = update_freq
        self.step_count = 0
        self.projection_matrices = {}  # U_k for each parameter
        
    def should_update_projection(self) -> bool:
        return self.step_count % self.update_freq == 0
    
    def update_projections(self, model):   
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
                
            grad = param.grad.data
            original_shape = grad.shape
            
            if len(original_shape) == 1:
                continue
            
            if len(original_shape) > 2:
                grad_2d = grad.reshape(-1, original_shape[-1])
            else:
                grad_2d = grad
            
            m, n = grad_2d.shape
            if min(m, n) < self.rank:
                continue
            
            # Only apply rSVD if matrix is large enough
            initial_rank = min(self.rank * 2, min(m, n))
            U, S, V = torch.svd_lowrank(
                grad_2d, 
                q=initial_rank
            )

            k = min(initial_rank, self.rank)    
            U_k = U[:, :k]
            S_k = S[:k]
            V_k = V[:, :k]
            
            self.projection_matrices[name] = {
                'U_k': U_k,      # (m, k)
                'S_k': S_k,      # (k,)
                'V_k': V_k,      # (n, k)
                'original_shape': original_shape,
            }
                
    
    def project_gradients(self, model):
        """
        2. Low-rank approximation: g_approx = U_k S_k V_k^T (uses compressed form)
        
        Option 2 uses the low-rank approximation directly.
        """
        for name, param in model.named_parameters():
            if name not in self.projection_matrices or param.grad is None:
                continue
            
            proj_info = self.projection_matrices[name]
            U_k = proj_info['U_k']
            S_k = proj_info['S_k']
            V_k = proj_info['V_k']
            original_shape = proj_info['original_shape']

            grad = param.grad.data
            if len(original_shape) > 2:
                grad_2d = grad.reshape(-1, original_shape[-1])
            else:
                grad_2d = grad

            grad_projected_2d = torch.matmul(U_k, torch.matmul(U_k.T, grad_2d))
            grad_projected = grad_projected_2d.reshape(original_shape)
            param.grad.data = grad_projected
            
    
    def step(self, model):
        if self.should_update_projection():
            self.update_projections(model)
        
        if self.projection_matrices:
            self.project_gradients(model)
        
        self.step_count += 1