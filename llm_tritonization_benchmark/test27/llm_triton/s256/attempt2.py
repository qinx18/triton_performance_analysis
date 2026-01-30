import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_value, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Create block of j indices starting from 1
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid_j * BLOCK_SIZE + j_offsets + 1
    
    # Mask for valid j indices (j must be >= 1 and < len_2d)
    j_mask = j_indices < len_2d
    
    # Load a[j-1] values
    a_prev_indices = j_indices - 1
    a_prev_vals = tl.load(a_ptr + a_prev_indices, mask=j_mask, other=0.0)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_vals = 1.0 - a_prev_vals
    
    # Store a[j] values
    tl.store(a_ptr + j_indices, a_vals, mask=j_mask)
    
    # Calculate 2D indices for aa and bb arrays
    aa_indices = j_indices * len_2d + i_value
    bb_indices = j_indices * len_2d + i_value
    
    # Load bb[j][i] and d[j] values
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    d_vals = tl.load(d_ptr + j_indices, mask=j_mask, other=0.0)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_vals + bb_vals * d_vals
    
    # Store aa[j][i] values
    tl.store(aa_ptr + aa_indices, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = 128
    
    # Process each i sequentially
    for i in range(len_2d):
        # Calculate grid size for j dimension (starting from j=1)
        num_j = len_2d - 1  # j goes from 1 to len_2d-1
        grid_size = triton.cdiv(num_j, BLOCK_SIZE)
        
        # Launch kernel for this i value
        s256_kernel[(grid_size,)](
            a, aa, bb, d, i, len_2d, BLOCK_SIZE
        )