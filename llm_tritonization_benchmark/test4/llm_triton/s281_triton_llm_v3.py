import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i] * c[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        bc_product = b_vals * c_vals
        
        # Load a[LEN_1D-i-1] (reverse access)
        reverse_offsets = n_elements - 1 - current_offsets
        reverse_mask = (current_offsets < n_elements) & (reverse_offsets >= 0)
        a_reverse = tl.load(a_ptr + reverse_offsets, mask=reverse_mask, other=0.0)
        
        # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
        x = a_reverse + bc_product
        
        # Store a[i] = x - 1.0
        tl.store(a_ptr + current_offsets, x - 1.0, mask=mask)
        
        # Store b[i] = x
        tl.store(b_ptr + current_offsets, x, mask=mask)

def s281_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    s281_kernel[(1,)](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )