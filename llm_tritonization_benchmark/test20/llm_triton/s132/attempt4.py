import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start + 1
    
    mask = offsets < N
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load c[1] (scalar)
    c_1 = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] values - need to check bounds for i-1
    prev_offsets = offsets - 1
    prev_mask = (prev_offsets >= 0) & mask
    aa_k_offsets = k * N + prev_offsets
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=prev_mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store to aa[j][i]
    aa_j_offsets = j * N + offsets
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s132_kernel[grid](aa, b, c, j, k, N, BLOCK_SIZE=BLOCK_SIZE)