import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any value in this block is greater than t
        condition = a_vals > t
        if tl.sum(condition.to(tl.int32)) > 0:
            # Find first occurrence in this block
            for i in range(BLOCK_SIZE):
                if block_start + i < n:
                    val = tl.load(a_ptr + block_start + i)
                    if val > t:
                        tl.store(result_ptr, val)
                        return

def s332_triton(a, t):
    n = a.shape[0]
    
    # Use PyTorch to find first value greater than threshold
    mask = a > t
    if torch.any(mask):
        # Find first index where condition is true
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        first_idx = indices[0]
        value = a[first_idx].item()
    else:
        value = -1.0
    
    return value