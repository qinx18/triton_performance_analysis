import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calculate which iteration this thread block handles
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process iterations in blocks
    for i_base in range(block_start, n, tl.num_programs(0) * BLOCK_SIZE):
        i_offsets = i_base + offsets
        mask = i_offsets < n
        
        # Calculate k = 2*i for each i
        k_offsets = 2 * i_offsets
        k_mask = k_offsets < (2 * n)
        
        # Load data
        d_vals = tl.load(d_ptr + i_offsets, mask=mask)
        b_vals = tl.load(b_ptr + k_offsets, mask=mask & k_mask)
        c_vals = tl.load(c_ptr + k_offsets, mask=mask & k_mask)
        
        # Compute a[i] = b[k] - d[i]
        a_vals = b_vals - d_vals
        
        # Store a[i]
        tl.store(a_ptr + i_offsets, a_vals, mask=mask)
        
        # Compute b[k] = a[i] + c[k]
        b_new_vals = a_vals + c_vals
        
        # Store b[k]
        tl.store(b_ptr + k_offsets, b_new_vals, mask=mask & k_mask)

def s128_triton(a, b, c, d):
    n = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s128_kernel[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)