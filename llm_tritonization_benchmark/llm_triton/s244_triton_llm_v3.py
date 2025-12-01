import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Pre-define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially due to WAR dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # First statement: a[i] = b[i] + c[i] * d[i]
        new_a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        
        # Second statement: b[i] = c[i] + b[i]
        new_b_vals = c_vals + b_vals
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)
        
        # Third statement: a[i+1] = b[i] + a[i+1] * d[i]
        # Need to handle i+1 indices carefully
        next_offsets = current_offsets + 1
        next_mask = next_offsets < n_elements
        
        if tl.sum(next_mask.to(tl.int32)) > 0:
            a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask)
            updated_a_next = new_b_vals + a_next_vals * d_vals
            tl.store(a_ptr + next_offsets, updated_a_next, mask=next_mask)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single thread block due to dependencies
    
    s244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )