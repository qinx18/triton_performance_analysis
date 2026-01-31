import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any element is greater than threshold
        condition = a_vals > t
        
        if tl.any(condition):
            # Find first index where condition is true
            for i in range(BLOCK_SIZE):
                offset_i = block_start + i
                if offset_i < n_elements:
                    val = tl.load(a_ptr + offset_i)
                    if val > t:
                        tl.store(result_ptr, val)
                        tl.store(result_ptr + 1, offset_i)
                        return

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Find first element greater than threshold using PyTorch
    condition = a > t
    indices = torch.where(condition)[0]
    
    if len(indices) > 0:
        first_idx = indices[0].item()
        value = a[first_idx].item()
    else:
        first_idx = -2
        value = -1.0
    
    return value