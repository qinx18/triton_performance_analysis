import triton
import triton.language as tl
import torch

@triton.jit
def s4117_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        i_offsets = block_start + offsets
        mask = i_offsets < n_elements
        
        # Compute c indices: i/2
        c_indices = i_offsets // 2
        
        # Load values
        b_vals = tl.load(b_ptr + i_offsets, mask=mask)
        c_vals = tl.load(c_ptr + c_indices, mask=mask)
        d_vals = tl.load(d_ptr + i_offsets, mask=mask)
        
        # Compute: a[i] = b[i] + c[i/2] * d[i]
        result = b_vals + c_vals * d_vals
        
        # Store result
        tl.store(a_ptr + i_offsets, result, mask=mask)

def s4117_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4117_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )