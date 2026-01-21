import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # i ranges from j+1 to N-1 (inclusive)
    i_start = j + 1
    mask = (offsets >= i_start) & (offsets < N)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load aa[j][i] values - correct 2D indexing
    aa_indices = j * N + offsets
    aa_ji = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Load a[j] (scalar)
    a_j = tl.load(a_ptr + j)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]  # LEN_2D
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    for j in range(N):
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)