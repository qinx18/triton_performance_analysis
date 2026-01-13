import torch
import triton
import triton.language as tl

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    # Each block processes elements sequentially
    block_start = block_id * BLOCK_SIZE
    
    if block_start >= n_elements:
        return
    
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Initialize t = 0 for this block
    t = 0.0
    
    # Process elements one by one sequentially
    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= block_end:
            return
            
        # Load b[i] and c[i]
        b_val = tl.load(b_ptr + idx)
        c_val = tl.load(c_ptr + idx)
        
        # s = b[i] * c[i]
        s = b_val * c_val
        
        # a[i] = s + t
        a_val = s + t
        tl.store(a_ptr + idx, a_val)
        
        # t = s for next iteration
        t = s

def s252_triton(a, b, c):
    n_elements = a.numel()
    
    # Use small block size since we need sequential processing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s252_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )