import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values for current iteration
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] from the read-only copy
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (block_start + BLOCK_SIZE + 1)
        next_mask = next_mask & (next_offsets < (a_copy_ptr.shape[0]))
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask, other=0.0)
        
        # First statement: a[i] *= c[i]
        a_new = a_vals * c_vals
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Second statement: b[i] += a[i + 1] * d[i]
        b_new = b_vals + a_next_vals * d_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )