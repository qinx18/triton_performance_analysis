import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel processes elements sequentially due to data dependencies
    # Each program processes one row at a time
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Process all valid j values for this i (j < i)
    max_j = i
    
    # Process in blocks along j dimension
    for j_start in range(0, max_j, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, max_j)
        block_size = j_end - j_start
        
        if block_size <= 0:
            break
            
        # Create offset vectors
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        j_mask = j_indices < max_j
        
        # Calculate memory addresses for aa[i][j] and aa[j][i]
        aa_ij_offsets = i * N + j_indices
        aa_ji_offsets = j_indices * N + i
        bb_ij_offsets = i * N + j_indices
        
        # Load aa[j][i] (transpose) and bb[i][j]
        aa_ji_vals = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask, other=0.0)
        bb_ij_vals = tl.load(bb_ptr + bb_ij_offsets, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[j][i] + bb[i][j]
        result = aa_ji_vals + bb_ij_vals
        
        # Store result to aa[i][j]
        tl.store(aa_ptr + aa_ij_offsets, result, mask=j_mask)

def s114_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Launch kernel with one program per row
    grid = (N,)
    
    s114_kernel[grid](
        aa, bb, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa