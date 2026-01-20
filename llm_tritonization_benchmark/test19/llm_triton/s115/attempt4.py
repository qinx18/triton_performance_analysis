import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE + (j + 1)
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Mask for valid indices (i < N)
    mask = i_indices < N
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    
    # Load aa[j][i] values (2D indexing: j * N + i)
    aa_ji_indices = j * N + i_indices
    aa_ji = tl.load(aa_ptr + aa_ji_indices, mask=mask, other=0.0)
    
    # Load a[j] (scalar)
    a_j = tl.load(a_ptr + j)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]  # LEN_2D dimension
    
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel computation for each i > j
    for j in range(N - 1):  # j goes from 0 to N-2 (since i must be > j)
        # Calculate grid size for processing i values from j+1 to N-1
        num_i_values = N - (j + 1)
        if num_i_values > 0:
            grid = (triton.cdiv(num_i_values, BLOCK_SIZE),)
            
            # Launch kernel
            s115_kernel[grid](
                a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE
            )