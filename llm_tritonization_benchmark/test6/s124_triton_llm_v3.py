import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate sequential output indices for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially within each block
    for i in range(BLOCK_SIZE):
        global_idx = block_start + i
        if global_idx >= n_elements:
            break
            
        # Load input values
        b_val = tl.load(b_ptr + global_idx)
        c_val = tl.load(c_ptr + global_idx)
        d_val = tl.load(d_ptr + global_idx)
        e_val = tl.load(e_ptr + global_idx)
        
        # Compute result based on condition
        if b_val > 0.0:
            result = b_val + d_val * e_val
        else:
            result = c_val + d_val * e_val
            
        # Store result at sequential output position
        output_idx = block_start + i
        tl.store(a_ptr + output_idx, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE,
    )