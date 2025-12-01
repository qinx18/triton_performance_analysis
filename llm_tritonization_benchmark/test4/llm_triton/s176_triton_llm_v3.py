import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, LEN_1D, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for j in range(m):  # j goes from 0 to LEN_1D/2 - 1
        c_j = tl.load(c_ptr + j)
        
        for block_start in range(0, m, BLOCK_SIZE):
            i_offsets = block_start + offsets
            mask = i_offsets < m
            
            # Load a[i]
            a_vals = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)
            
            # Calculate b index: i + m - j - 1
            b_indices = i_offsets + m - j - 1
            b_vals = tl.load(b_ptr + b_indices, mask=mask, other=0.0)
            
            # Compute a[i] += b[i+m-j-1] * c[j]
            result = a_vals + b_vals * c_j
            
            # Store back to a[i]
            tl.store(a_ptr + i_offsets, result, mask=mask)

def s176_triton(a, b, c):
    LEN_1D = a.shape[0]
    m = LEN_1D // 2
    
    BLOCK_SIZE = 256
    
    s176_kernel[(1,)](
        a, b, c,
        m, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )