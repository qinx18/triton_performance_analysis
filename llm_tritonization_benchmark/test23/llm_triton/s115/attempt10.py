import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Only process i values where i > j (since loop is for i in range(j+1, N))
    mask = (i_indices < N) & (i_indices >= j + 1)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    
    # Load a[j] (scalar)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values - aa is row-major, so aa[j][i] = aa[j*N + i]
    aa_ji_indices = j * N + i_indices
    aa_ji = tl.load(aa_ptr + aa_ji_indices, mask=mask, other=0.0)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(N):
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE)