import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Condition masks
        mask_neg = (d_vals < 0.0) & mask
        mask_zero = (d_vals == 0.0) & mask
        mask_pos = (d_vals > 0.0) & mask
        
        # Compute updates
        update_neg = b_vals * c_vals
        update_zero = b_vals * b_vals
        update_pos = c_vals * c_vals
        
        # Apply conditional updates
        result = a_vals
        result = tl.where(mask_neg, result + update_neg, result)
        result = tl.where(mask_zero, result + update_zero, result)
        result = tl.where(mask_pos, result + update_pos, result)
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    s441_kernel[(1,)](a, b, c, d, n_elements, BLOCK_SIZE)