import triton
import triton.language as tl
import torch

@triton.jit
def s176_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    len_1d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m
    
    # Load current values from the read-only copy
    a_vals = tl.load(a_copy_ptr + offsets, mask=mask, other=0.0)
    
    # Compute convolution for all j values
    for j in range(m):  # m = LEN_1D/2
        # Load c[j] (scalar, broadcast to all elements)
        c_val = tl.load(c_ptr + j)
        
        # Compute b indices: i + m - j - 1
        b_offsets = offsets + m - j - 1
        b_mask = mask & (b_offsets >= 0) & (b_offsets < len_1d)
        
        # Load b values
        b_vals = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        a_vals += b_vals * c_val
    
    # Store results back to original array
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s176_triton(a, b, c):
    len_1d = a.shape[0]
    m = len_1d // 2
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE']),)
    
    s176_kernel[grid](
        a,
        a_copy,
        b,
        c,
        m,
        len_1d,
        BLOCK_SIZE=BLOCK_SIZE,
    )