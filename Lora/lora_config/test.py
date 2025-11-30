import numpy as np
import torch

class SomeClass():
    
    def __init__(self):
        
        self.tensor = torch.tensor( np.ones( (500, 500)) , dtype=torch.int16, device="cuda")
        
        
    def run_epoch(self):
        #allocate some memory
        A = torch.tensor( np.ones(  (500, 500) ), dtype=torch.int16, device="cuda")
        #once we exict this method A should be freed 
        



C = SomeClass()
for i in range(10):
    torch.cuda.synchronize() 
    print("start epoch ---",i)
    perm_mem = torch.cuda.memory_allocated()
    print("perm_mem",perm_mem)
    torch.cuda.reset_peak_memory_stats()
    print("memory reset!")
    peak2 = torch.cuda.max_memory_allocated()
    print("peak2 ", peak2)

    
    C.run_epoch()
    
    peak = torch.cuda.max_memory_allocated()
    print("peak memory", peak)
    print("total memory consumed", peak + perm_mem)
    