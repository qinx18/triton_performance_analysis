import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + 1
    
    mask = i_offsets < N
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load c[1] (scalar broadcast)
    c_1 = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] values
    aa_k_offsets = k * N + (i_offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store to aa[j][i]
    aa_j_offsets = j * N + i_offsets
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k, N, BLOCK_SIZE
    )