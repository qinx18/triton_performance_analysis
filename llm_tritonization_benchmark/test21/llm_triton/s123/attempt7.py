import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    # Each thread handles one element
    i = tl.program_id(0)
    
    if i >= n_half:
        return
    
    # Calculate j by counting increments from all previous iterations
    j = i  # Base increment (one per iteration)
    
    # Add extra increments from previous iterations where c[k] > 0
    extra = 0
    k_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks to count previous c[k] > 0 conditions
    for k_block_start in range(0, i, BLOCK_SIZE):
        k_block_offsets = k_block_start + k_offsets
        k_mask = k_block_offsets < i
        
        if k_block_start < i:
            c_vals = tl.load(c_ptr + k_block_offsets, mask=k_mask, other=0.0)
            extra += tl.sum(tl.where(c_vals > 0.0, 1, 0))
    
    j = j + extra
    
    # Load values for current i
    b_i = tl.load(b_ptr + i)
    c_i = tl.load(c_ptr + i)
    d_i = tl.load(d_ptr + i)
    e_i = tl.load(e_ptr + i)
    
    # First store: a[j] = b[i] + d[i] * e[i]
    val1 = b_i + d_i * e_i
    tl.store(a_ptr + j, val1)
    
    # Check condition and second store if needed
    if c_i > 0.0:
        j = j + 1
        val2 = c_i + d_i * e_i
        tl.store(a_ptr + j, val2)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    # Use one thread per element for simplicity
    grid = (n_half,)
    BLOCK_SIZE = 256
    
    s123_kernel[grid](a, b, c, d, e, n_half, BLOCK_SIZE=BLOCK_SIZE)