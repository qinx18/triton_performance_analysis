import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(aa_ptr, bb_ptr, a_val, d_val, j, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID and create block of i indices
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = pid * BLOCK_SIZE + i_offsets
    
    # Mask for valid i indices
    i_mask = i_indices < len_2d
    
    # Calculate 2D indices for aa[j][i] and bb[j][i]
    indices_2d = j * len_2d + i_indices
    
    # Load bb[j][i] values
    bb_vals = tl.load(bb_ptr + indices_2d, mask=i_mask, other=0.0)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i] values
    tl.store(aa_ptr + indices_2d, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = 128
    
    # Process each i sequentially
    for i in range(len_2d):
        # Process each j sequentially (j dependency requires this)
        for j in range(1, len_2d):
            # Update a[j] = 1.0 - a[j-1]
            a[j] = 1.0 - a[j-1]
            
            # Calculate grid size for i dimension
            grid_size = triton.cdiv(len_2d, BLOCK_SIZE)
            
            # Launch kernel for this j value, parallelizing over i
            s256_kernel[(grid_size,)](
                aa,
                bb,
                a[j].item(),
                d[j].item(),
                j,
                len_2d, 
                BLOCK_SIZE
            )