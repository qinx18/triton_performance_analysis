import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process each element in the block sequentially to handle dependencies
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements - 1:
            break
            
        # Load b[idx] + e[idx] for a[idx+1]
        b_val = tl.load(b_ptr + idx)
        e_val = tl.load(e_ptr + idx)
        temp1 = b_val + e_val
        
        # Load b[idx] + c[idx] for a[idx] 
        c_val = tl.load(c_ptr + idx)
        temp2 = b_val + c_val
        
        # Store a[idx+1] = b[idx] + e[idx]
        tl.store(a_ptr + idx + 1, temp1)
        
        # Store a[idx] = b[idx] + c[idx]
        tl.store(a_ptr + idx, temp2)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )