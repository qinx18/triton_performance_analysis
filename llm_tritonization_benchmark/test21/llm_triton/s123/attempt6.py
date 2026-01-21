import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load input data
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Calculate j values: j starts at 0 for i=0, then increments
    # For each i, we write to j, and if c[i] > 0, we also write to j+1
    # So j = 2*i if all previous c values were > 0
    # We need to compute the actual j values considering the condition
    
    for idx in range(BLOCK_SIZE):
        if block_start + idx >= n_half:
            break
            
        i = block_start + idx
        # Calculate actual j value by counting previous increments
        j = i  # Start with base increment (one j++ per iteration)
        
        # Add extra increments from previous iterations where c[k] > 0
        for k in range(i):
            c_k = tl.load(c_ptr + k)
            if c_k > 0.0:
                j += 1
        
        # First store: a[j] = b[i] + d[i] * e[i]
        val1 = tl.load(b_ptr + i) + tl.load(d_ptr + i) * tl.load(e_ptr + i)
        tl.store(a_ptr + j, val1)
        
        # Check condition and second store if needed
        c_i = tl.load(c_ptr + i)
        if c_i > 0.0:
            j += 1
            val2 = c_i + tl.load(d_ptr + i) * tl.load(e_ptr + i)
            tl.store(a_ptr + j, val2)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](a, b, c, d, e, n_half, BLOCK_SIZE=BLOCK_SIZE)