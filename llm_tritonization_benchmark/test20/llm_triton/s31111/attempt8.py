import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        sum_val = 0.0
        
        # Process 8 groups of 4 elements each, starting at indices 0,4,8,12,16,20,24,28
        for group_idx in range(8):
            start_idx = group_idx * 4
            
            # Check if we have at least one valid element in this group
            if start_idx < N:
                # Load 4 elements starting at start_idx
                offsets = start_idx + tl.arange(0, 4)
                mask = offsets < N
                vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
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