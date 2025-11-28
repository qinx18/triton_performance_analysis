import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, N, j, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid = tl.program_id(0)
    
    # Calculate starting i index (j+1 since i > j)
    i_start = j + 1
    i_offset = pid * BLOCK_SIZE
    i_idx = i_start + i_offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid i indices (must be < N and > j)
    mask = i_idx < N
    
    # Load a[j] (broadcast value)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j, i] values
    aa_ji = tl.load(aa_ptr + j * N + i_idx, mask=mask)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_idx, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    a_i_new = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_idx, a_i_new, mask=mask)

def s115_triton(a, aa):
    a = a.contiguous()
    aa = aa.contiguous()
    
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j dimension (required for correctness)
    for j in range(LEN_2D - 1):  # j goes from 0 to LEN_2D-2
        num_i = LEN_2D - (j + 1)  # Number of i iterations for this j
        if num_i > 0:
            grid = (triton.cdiv(num_i, BLOCK_SIZE),)
            s115_kernel[grid](a, aa, LEN_2D, j, BLOCK_SIZE=BLOCK_SIZE)
    
    return a