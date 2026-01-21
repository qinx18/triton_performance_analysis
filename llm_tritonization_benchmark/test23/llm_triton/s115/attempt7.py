import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Only process i values where i > j
    mask = (i_indices < N) & (i_indices > j)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
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
        # Calculate grid size for parallelizing over i values
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE)