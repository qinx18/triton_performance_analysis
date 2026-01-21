import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    mask = i_indices < N
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask)
    
    # Load aa[j][i] values
    aa_ji = tl.load(aa_ptr + j * N + i_indices, mask=mask)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(N):
        num_elements = N
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)