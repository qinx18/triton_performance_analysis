import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid_i = tl.program_id(0)
    
    # Calculate i index
    i = pid_i
    
    # Check bounds for i
    if i >= len_2d:
        return
    
    # Create offset vector for vectorized operations
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks of j
    for j_start in range(1, len_2d, BLOCK_SIZE):
        current_j_offsets = j_start + j_offsets
        mask = (current_j_offsets < len_2d) & (current_j_offsets >= 1)
        
        # Load a[j-1] values
        a_prev_offsets = current_j_offsets - 1
        a_prev_mask = (a_prev_offsets >= 0) & (a_prev_offsets < len_2d)
        a_prev_vals = tl.load(a_ptr + a_prev_offsets, mask=a_prev_mask, other=0.0)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_vals = 1.0 - a_prev_vals
        
        # Store a[j] values
        tl.store(a_ptr + current_j_offsets, a_vals, mask=mask)
        
        # Calculate 2D indices for aa and bb
        aa_indices = current_j_offsets * len_2d + i
        bb_indices = current_j_offsets * len_2d + i
        
        # Load bb[j][i] and d[j] values
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_j_offsets, mask=mask, other=0.0)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_vals + bb_vals * d_vals
        
        # Store aa[j][i] values
        tl.store(aa_ptr + aa_indices, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d, len_2d):
    # Process each j sequentially to handle the dependency
    for j in range(1, len_2d):
        # Compute a[j] = 1.0 - a[j-1]
        a[j] = 1.0 - a[j-1]
        
        # Parallelize over all i values for aa[j][i] = a[j] + bb[j][i] * d[j]
        aa[j, :] = a[j] + bb[j, :] * d[j]