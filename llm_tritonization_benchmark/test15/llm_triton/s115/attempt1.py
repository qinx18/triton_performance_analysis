import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Calculate starting index for this j value
    start_i = j + 1
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[j] once (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Process all valid i values in blocks
    for block_start in range(start_i, LEN_2D, BLOCK_SIZE):
        i_offsets = block_start + offsets
        
        # Mask for valid indices
        mask = i_offsets < LEN_2D
        
        # Load a[i] values
        a_vals = tl.load(a_ptr + i_offsets, mask=mask)
        
        # Load aa[j][i] values (row-major: aa[j][i] = aa[j*LEN_2D + i])
        aa_indices = j * LEN_2D + i_offsets
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask)
        
        # Compute: a[i] -= aa[j][i] * a[j]
        result = a_vals - aa_vals * a_j
        
        # Store back to a[i]
        tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j values
    for j in range(LEN_2D):
        # Launch kernel for this j value
        grid = (1,)
        s115_kernel[grid](
            a, aa, j, LEN_2D, BLOCK_SIZE
        )