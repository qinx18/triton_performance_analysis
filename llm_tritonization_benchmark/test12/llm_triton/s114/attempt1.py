import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the lower triangular matrix sequentially
    # due to data dependencies (aa[i][j] = aa[j][i] + bb[i][j])
    
    for i in range(LEN_2D):
        # For each row i, process columns j where j < i
        if i == 0:
            continue  # No valid j for i=0
            
        # Process elements in blocks for the valid range [0, i)
        for j_start in range(0, i, BLOCK_SIZE):
            j_offsets = tl.arange(0, BLOCK_SIZE)
            j_indices = j_start + j_offsets
            
            # Mask to ensure j < i and j >= 0
            mask = (j_indices < i) & (j_indices >= 0)
            
            if tl.sum(mask.to(tl.int32)) > 0:  # Only proceed if there are valid elements
                # Calculate flat indices for aa[i][j] and bb[i][j]
                aa_ij_indices = i * LEN_2D + j_indices
                bb_ij_indices = i * LEN_2D + j_indices
                
                # Calculate flat indices for aa[j][i] (transpose access)
                aa_ji_indices = j_indices * LEN_2D + i
                
                # Load aa[j][i] and bb[i][j]
                aa_ji_vals = tl.load(aa_ptr + aa_ji_indices, mask=mask, other=0.0)
                bb_ij_vals = tl.load(bb_ptr + bb_ij_indices, mask=mask, other=0.0)
                
                # Compute aa[i][j] = aa[j][i] + bb[i][j]
                result = aa_ji_vals + bb_ij_vals
                
                # Store result to aa[i][j]
                tl.store(aa_ptr + aa_ij_indices, result, mask=mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Ensure tensors are contiguous
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    # Launch kernel with single thread block since we need sequential processing
    s114_kernel[(1,)](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa