import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        sum_val = 0.0
        
        # Process 8 groups: a[0:4], a[4:8], a[8:12], ..., a[28:32]
        for group_idx in range(8):
            start_idx = group_idx * 4
            
            # Load 4 elements starting from start_idx
            offsets = tl.arange(0, 4) + start_idx
            mask = offsets < N
            vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
            
            # Sum the 4 elements
            group_sum = tl.sum(vals)
            sum_val += group_sum
        
        tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_tensor, N
    )
    
    return sum_tensor.item()