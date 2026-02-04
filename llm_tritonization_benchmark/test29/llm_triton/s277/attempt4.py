import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements - 1)
    
    # Process elements sequentially within each block
    for idx in range(block_start, block_end):
        # Load current values
        a_val = tl.load(a_ptr + idx)
        b_val = tl.load(b_ptr + idx)
        c_val = tl.load(c_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        e_val = tl.load(e_ptr + idx)
        
        # Check first condition: if (a[i] >= 0.) goto L20
        a_condition = a_val >= 0.0
        
        # Only proceed if a[i] < 0
        b_condition = b_val >= 0.0
        
        # Update a[i] only if both conditions are false
        should_update_a = (a_condition == False) & (b_condition == False)
        new_a_val = tl.where(should_update_a, a_val + c_val * d_val, a_val)
        tl.store(a_ptr + idx, new_a_val)
        
        # Update b[i+1] if a[i] < 0 (either b condition true or false)
        should_update_b = a_condition == False
        new_b_val = c_val + d_val * e_val
        
        if should_update_b:
            tl.store(b_ptr + idx + 1, new_b_val)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )