import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load values for a[i] computation
        b_prev = tl.load(b_copy_ptr + current_offsets - 1, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load values for b[i] computation  
        b_next = tl.load(b_copy_ptr + current_offsets + 1, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Compute results
        a_result = b_prev + c_vals * d_vals
        b_result = b_next - e_vals * d_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, a_result, mask=mask)
        tl.store(b_ptr + current_offsets, b_result, mask=mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy of b to handle WAR dependency
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )