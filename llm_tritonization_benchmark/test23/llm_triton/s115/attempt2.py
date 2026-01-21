import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this block
    pid = tl.program_id(0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Create mask for valid indices (i >= j+1 and i < N)
    mask = (i_indices >= j + 1) & (i_indices < N)
    
    # Load a[i] values
    a_i_ptrs = a_ptr + i_indices
    a_i = tl.load(a_i_ptrs, mask=mask, other=0.0)
    
    # Load aa[j][i] values
    aa_ji_ptrs = aa_ptr + j * N + i_indices
    aa_ji = tl.load(aa_ji_ptrs, mask=mask, other=0.0)
    
    # Load a[j] (scalar)
    a_j = tl.load(a_ptr + j)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store result back to a[i]
    tl.store(a_i_ptrs, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]  # LEN_2D
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel over i for each j
    for j in range(N):
        # Launch kernel with enough blocks to cover all i values
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)