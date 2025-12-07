import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid i indices (i starts from j+1)
    mask = i_offsets < (LEN_2D - j - 1)
    
    # Adjust i_offsets to actual i values (i = j + 1 + offset)
    actual_i = i_offsets + j + 1
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * LEN_2D + actual_i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i] values
    a_i_vals = tl.load(a_ptr + actual_i, mask=mask)
    
    # Compute: a[i] -= aa[j][i] * a[j]
    result = a_i_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + actual_i, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(LEN_2D):
        # Number of i elements to process (i goes from j+1 to LEN_2D-1)
        num_i = LEN_2D - (j + 1)
        if num_i <= 0:
            continue
            
        # Launch kernel for this j
        grid = (triton.cdiv(num_i, BLOCK_SIZE),)
        s115_kernel[grid](
            a, aa, j, LEN_2D, BLOCK_SIZE
        )