import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for blocks of i indices
    pid = tl.program_id(0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Create mask for valid i indices (must be >= j+1 and < len_2d)
    mask = (i_indices >= j + 1) & (i_indices < len_2d)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_indices, mask=mask)
    
    # Load aa[j][i] values (2D indexing: j * len_2d + i)
    aa_indices = j * len_2d + i_indices
    aa_ji = tl.load(aa_ptr + aa_indices, mask=mask)
    
    # Load a[j] (scalar, broadcast to vector)
    a_j = tl.load(a_ptr + j)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store result back to a[i]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa, len_2d):
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(len_2d):
        # Calculate grid size for i indices (j+1 to len_2d-1)
        num_i = len_2d - (j + 1)
        if num_i <= 0:
            continue
            
        grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
        
        # Launch kernel for this j
        s115_kernel[grid](
            a, aa, j, len_2d, 
            BLOCK_SIZE=BLOCK_SIZE
        )