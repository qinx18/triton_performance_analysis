import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program id
    pid = tl.program_id(0)
    
    # Calculate offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Adjust i_offsets to start from j+1
    actual_i = (j + 1) + i_offsets
    
    # Mask for valid i indices (actual_i < len_2d)
    mask = actual_i < len_2d
    
    # Load a[i] values
    a_i = tl.load(a_ptr + actual_i, mask=mask, other=0.0)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_indices = j * len_2d + actual_i
    aa_ji = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + actual_i, result, mask=mask)

def s115_triton(a, aa, len_2d):
    # Sequential loop over j
    for j in range(len_2d):
        # Only launch kernel if there are valid i values (i > j)
        if j + 1 < len_2d:
            BLOCK_SIZE = 128
            num_valid_i = len_2d - (j + 1)
            grid = (triton.cdiv(num_valid_i, BLOCK_SIZE),)
            
            s115_kernel[grid](a, aa, j, len_2d, BLOCK_SIZE)