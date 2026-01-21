import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    # Get block of indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + 1  # Start from i=1
    
    # Create mask for valid indices (i from 1 to N-1)
    mask = i_indices < N
    
    # Load b[i] values
    b_ptrs = b_ptr + i_indices
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
    
    # Load c[1] (scalar)
    c_1 = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] values - need to handle i-1 properly
    i_minus_1 = i_indices - 1
    aa_k_ptrs = aa_ptr + k * N + i_minus_1
    aa_k_vals = tl.load(aa_k_ptrs, mask=mask, other=0.0)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store to aa[j][i]
    aa_j_ptrs = aa_ptr + j * N + i_indices
    tl.store(aa_j_ptrs, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)  # N-1 because we start from i=1
    
    s132_kernel[grid](
        aa, b, c, j, k, N,
        BLOCK_SIZE=BLOCK_SIZE
    )