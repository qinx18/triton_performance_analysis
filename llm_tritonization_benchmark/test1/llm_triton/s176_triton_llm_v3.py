import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(
    a_ptr, b_ptr, c_ptr,
    LEN_1D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which j iteration this block handles
    j = pid
    
    if j >= LEN_1D // 2:
        return
    
    # Load c[j] once for this j iteration
    c_j = tl.load(c_ptr + j)
    
    # Calculate m = LEN_1D/2
    m = LEN_1D // 2
    
    # Process all i values for this j
    for i_start in range(0, m, BLOCK_SIZE):
        # Calculate offsets for this block of i values
        i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
        i_mask = i_offsets < m
        
        # Load a[i] values
        a_vals = tl.load(a_ptr + i_offsets, mask=i_mask, other=0.0)
        
        # Calculate b indices: i + m - j - 1
        b_indices = i_offsets + m - j - 1
        b_vals = tl.load(b_ptr + b_indices, mask=i_mask, other=0.0)
        
        # Compute a[i] += b[i+m-j-1] * c[j]
        result = a_vals + b_vals * c_j
        
        # Store back to a[i]
        tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s176_triton(a, b, c):
    LEN_1D = a.shape[0]
    
    # Use j dimension for parallelization (LEN_1D/2 blocks)
    grid = (LEN_1D // 2,)
    BLOCK_SIZE = 256
    
    s176_kernel[grid](
        a, b, c,
        LEN_1D=LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )