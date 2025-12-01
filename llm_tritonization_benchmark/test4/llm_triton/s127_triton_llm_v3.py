import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process iterations in blocks
    for i_block in range(block_start, n, tl.num_programs(0) * BLOCK_SIZE):
        i_offsets = i_block + offsets
        mask = i_offsets < n
        
        # Load b[i], c[i], d[i], e[i]
        b_vals = tl.load(b_ptr + i_offsets, mask=mask)
        c_vals = tl.load(c_ptr + i_offsets, mask=mask)
        d_vals = tl.load(d_ptr + i_offsets, mask=mask)
        e_vals = tl.load(e_ptr + i_offsets, mask=mask)
        
        # Compute j indices: j = 2*i and j = 2*i+1
        j_even = 2 * i_offsets
        j_odd = 2 * i_offsets + 1
        
        # Compute values
        a_even = b_vals + c_vals * d_vals
        a_odd = b_vals + d_vals * e_vals
        
        # Store results
        tl.store(a_ptr + j_even, a_even, mask=mask)
        tl.store(a_ptr + j_odd, a_odd, mask=mask)

def s127_triton(a, b, c, d, e):
    n = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )