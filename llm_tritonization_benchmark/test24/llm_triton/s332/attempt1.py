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
        
        condition = a_vals > t
        if tl.any(condition):
            # Find first index where condition is true
            for i in range(BLOCK_SIZE):
                offset = block_start + i
                if offset < n_elements:
                    val = tl.load(a_ptr + offset)
                    if val > t:
                        tl.store(result_ptr, val)
                        return

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use PyTorch for efficient first-match finding
    mask = a > t
    if torch.any(mask):
        # Find first index where condition is true
        first_idx = torch.argmax(mask.int())
        value = a[first_idx].item()
        return value
    else:
        return -1.0