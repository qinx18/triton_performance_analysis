import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for idx in range(BLOCK_SIZE):
        i = block_start + idx
        mask_i = i < n_elements
        
        if not mask_i:
            return
            
        k = 2 * i
        
        d_val = tl.load(d_ptr + i, mask=mask_i)
        b_val = tl.load(b_ptr + k)
        c_val = tl.load(c_ptr + k)
        
        a_val = b_val - d_val
        tl.store(a_ptr + i, a_val, mask=mask_i)
        
        b_new_val = a_val + c_val
        tl.store(b_ptr + k, b_new_val)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 64
    
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s128_kernel[(grid_size,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b