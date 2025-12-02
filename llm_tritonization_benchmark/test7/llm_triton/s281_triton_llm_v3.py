import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < n_elements
        
        # Compute reverse indices for a[LEN_1D-i-1]
        reverse_offsets = (n_elements - 1) - current_offsets
        reverse_mask = (reverse_offsets >= 0) & (reverse_offsets < n_elements) & mask
        
        # Load values
        a_reverse = tl.load(a_ptr + reverse_offsets, mask=reverse_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
        x_vals = a_reverse + b_vals * c_vals
        
        # Store x values to x array for output
        tl.store(x_ptr + current_offsets, x_vals, mask=mask)
        
        # Update arrays
        # a[i] = x - 1.0
        a_new = x_vals - 1.0
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # b[i] = x
        tl.store(b_ptr + current_offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)  # Single block processes all elements sequentially
    
    s281_kernel[grid](
        a, b, c, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )