import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    j = 0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        de_product = d_vals * e_vals
        condition = b_vals > 0.0
        
        result = tl.where(condition, b_vals + de_product, c_vals + de_product)
        
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                tl.store(a_ptr + j, tl.load(a_ptr + block_start + i, mask=(block_start + i < n_elements)) if condition.to(tl.int1)[i] else result[i])
                j += 1

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)
    s124_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)