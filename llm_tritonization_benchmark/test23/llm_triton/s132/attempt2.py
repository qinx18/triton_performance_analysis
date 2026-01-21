import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Loop indices: i from 1 to N-1
    i_offsets = offsets + 1
    mask = i_offsets < N
    
    # Load b[i]
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load c[1] (scalar broadcast)
    c_1 = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] = aa[k*N + (i-1)]
    aa_k_indices = k * N + i_offsets - 1
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store to aa[j][i] = aa[j*N + i]
    aa_j_indices = j * N + i_offsets
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    N = aa.shape[0]
    
    # m = 0, so j = 0 and k = 1 by default, but use passed values
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k, N,
        BLOCK_SIZE=BLOCK_SIZE
    )