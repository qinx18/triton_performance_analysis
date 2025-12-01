import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially to maintain dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current elements
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # First statement: a[i] *= c[i]
        a_vals = a_vals * c_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Load a[i+1] for second statement
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n_elements + 1)  # Allow reading a[n_elements-1+1]
        a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask)
        
        # Second statement: b[i] += a[i + 1] * d[i]
        # Only update where we have valid next elements
        valid_mask = mask & (current_offsets + 1 < (n_elements + 1))
        b_update = a_next_vals * d_vals
        b_vals = tl.where(valid_mask, b_vals + b_update, b_vals)
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single thread block to maintain sequential execution
    
    s212_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )