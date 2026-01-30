import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, len_2d, j, k, BLOCK_SIZE: tl.constexpr):
    # Get starting position for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector (define once, not in loop)
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + 1  # Start from i=1
    
    # Create mask for valid indices
    mask = (i_offsets >= 1) & (i_offsets < len_2d)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    
    # Load c[1] (scalar broadcast)
    c1_val = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] values
    aa_k_indices = k * len_2d + (i_offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask, other=0.0)
    
    # Compute: aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1_val
    
    # Store to aa[j][i]
    aa_j_indices = j * len_2d + i_offsets
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, len_2d, j, k):
    # Get dimensions
    N = len_2d - 1  # Process indices 1 to len_2d-1
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    s132_kernel[grid](
        aa, b, c, len_2d, j, k,
        BLOCK_SIZE=BLOCK_SIZE
    )