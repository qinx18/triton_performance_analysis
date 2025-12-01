import triton
import triton.language as tl
import torch

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, LEN_1D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one value of j
    j = pid
    
    if j >= m:  # m = LEN_1D/2
        return
    
    # Load c[j] once for this j
    c_j = tl.load(c_ptr + j)
    
    # Process all i values for this j
    for i_start in range(0, m, BLOCK_SIZE):
        i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
        mask = i_offsets < m
        
        # Load a[i]
        a_vals = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)
        
        # Calculate b index: i + m - j - 1
        b_indices = i_offsets + m - j - 1
        # Load b[i+m-j-1]
        b_vals = tl.load(b_ptr + b_indices, mask=mask, other=0.0)
        
        # Compute a[i] += b[i+m-j-1] * c[j]
        result = a_vals + b_vals * c_j
        
        # Store result back to a[i]
        tl.store(a_ptr + i_offsets, result, mask=mask)

def s176_triton(a, b, c):
    LEN_1D = a.shape[0]
    m = LEN_1D // 2
    
    BLOCK_SIZE = 256
    grid = (m,)
    
    s176_kernel[grid](
        a, b, c,
        m, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a