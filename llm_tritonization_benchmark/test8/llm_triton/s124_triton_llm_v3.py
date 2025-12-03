import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process all elements sequentially within this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Load single elements
        b_val = tl.load(b_ptr + idx)
        d_val = tl.load(d_ptr + idx) 
        e_val = tl.load(e_ptr + idx)
        
        # Conditional computation
        if b_val > 0.0:
            result = b_val + d_val * e_val
        else:
            c_val = tl.load(c_ptr + idx)
            result = c_val + d_val * e_val
            
        # Store result at same position (j = i since j starts at -1 and increments)
        tl.store(a_ptr + idx, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)