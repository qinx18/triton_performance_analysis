import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one j iteration
    j = pid
    
    # Compute for all i values in this j iteration
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_start in range(0, m, BLOCK_SIZE):
        i_offsets = i_start + offsets
        i_mask = i_offsets < m
        
        # Load a[i]
        a_vals = tl.load(a_ptr + i_offsets, mask=i_mask)
        
        # Load b[i+m-j-1]
        b_indices = i_offsets + m - j - 1
        b_vals = tl.load(b_ptr + b_indices, mask=i_mask)
        
        # Load c[j] (scalar broadcast)
        c_val = tl.load(c_ptr + j)
        
        # Compute a[i] += b[i+m-j-1] * c[j]
        result = a_vals + b_vals * c_val
        
        # Store back to a[i]
        tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s176_triton(a, b, c, m):
    BLOCK_SIZE = 256
    grid = (m,)
    
    s176_kernel[grid](a, b, c, m, BLOCK_SIZE=BLOCK_SIZE)