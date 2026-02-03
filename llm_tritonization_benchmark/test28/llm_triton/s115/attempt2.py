import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid = tl.program_id(0)
    
    # Calculate i offsets for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    # Sequential loop over j inside kernel
    for j in range(len_2d - 1):
        # Only process i values where i > j and i < len_2d
        i_mask = (i_idx > j) & (i_idx < len_2d)
        
        # Load a[j] (scalar, broadcast to vector)
        a_j = tl.load(a_ptr + j)
        
        # Load a[i] values for valid indices
        a_i = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
        
        # Load aa[j][i] values
        aa_offsets = j * len_2d + i_idx
        aa_ji = tl.load(aa_ptr + aa_offsets, mask=i_mask, other=0.0)
        
        # Compute: a[i] -= aa[j][i] * a[j]
        result = a_i - aa_ji * a_j
        
        # Store back to a[i]
        tl.store(a_ptr + i_idx, result, mask=i_mask)

def s115_triton(a, aa, len_2d):
    BLOCK_SIZE = 256
    
    # Calculate grid size for i dimension
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    # Launch kernel
    s115_kernel[grid](
        a, aa, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )