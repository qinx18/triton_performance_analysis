import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    # Get the starting position for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets + 1  # +1 because loop starts at i=1
    
    # Create mask for valid indices
    mask = current_offsets < N
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Load c[1] (scalar)
    c1_val = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] values - note i-1 indexing
    aa_k_offsets = k * N + (current_offsets - 1)
    # Create mask for aa[k][i-1] to ensure i-1 >= 0
    aa_k_mask = mask & (current_offsets >= 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=aa_k_mask, other=0.0)
    
    # Compute: aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1_val
    
    # Store to aa[j][i]
    aa_j_offsets = j * N + current_offsets
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    # Grid and block size configuration
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(N - 1, BLOCK_SIZE)  # N-1 because we start from i=1
    
    # Launch kernel
    s132_kernel[(grid_size,)](
        aa, b, c, j, k, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa