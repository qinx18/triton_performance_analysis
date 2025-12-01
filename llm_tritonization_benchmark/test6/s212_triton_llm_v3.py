import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current elements
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] values (need to handle bounds)
        next_offsets = current_offsets + 1
        next_mask = mask & (next_offsets < (n_elements + 1))
        a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask)
        
        # Perform computations: a[i] *= c[i]
        new_a_vals = a_vals * c_vals
        
        # Store updated a values
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        
        # Perform computations: b[i] += a[i + 1] * d[i]
        # Use the original a[i+1] values (before update)
        new_b_vals = b_vals + a_next_vals * d_vals
        
        # Store updated b values
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    BLOCK_SIZE = 256
    
    s212_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )