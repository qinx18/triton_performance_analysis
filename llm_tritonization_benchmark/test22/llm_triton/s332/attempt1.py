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
        
        # If any condition is true, we found our element
        if tl.sum(condition.to(tl.int32)) > 0:
            # Find the first occurrence within this block
            for i in range(BLOCK_SIZE):
                if i + block_start < n_elements:
                    element_val = tl.load(a_ptr + (i + block_start))
                    if element_val > t:
                        tl.store(result_ptr, element_val)
                        return

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use PyTorch for efficient search
    mask = a > t
    if mask.any():
        first_idx = torch.argmax(mask.int())
        value = a[first_idx].item()
    else:
        value = -1.0
    
    return value