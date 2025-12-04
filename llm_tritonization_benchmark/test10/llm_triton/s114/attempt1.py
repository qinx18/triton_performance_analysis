import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential implementation due to data dependencies
    for i in range(LEN_2D):
        # Process elements in row i where j < i
        num_j = i  # j goes from 0 to i-1
        if num_j > 0:
            for j_start in range(0, num_j, BLOCK_SIZE):
                j_offsets = tl.arange(0, BLOCK_SIZE)
                j_indices = j_start + j_offsets
                j_mask = j_indices < num_j
                
                # Calculate flat indices for aa[i][j]
                aa_ij_indices = i * LEN_2D + j_indices
                # Calculate flat indices for aa[j][i] 
                aa_ji_indices = j_indices * LEN_2D + i
                # Calculate flat indices for bb[i][j]
                bb_ij_indices = i * LEN_2D + j_indices
                
                # Load values
                aa_ji_vals = tl.load(aa_ptr + aa_ji_indices, mask=j_mask, other=0.0)
                bb_ij_vals = tl.load(bb_ptr + bb_ij_indices, mask=j_mask, other=0.0)
                
                # Compute aa[i][j] = aa[j][i] + bb[i][j]
                result = aa_ji_vals + bb_ij_vals
                
                # Store result
                tl.store(aa_ptr + aa_ij_indices, result, mask=j_mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Launch kernel with single program
    s114_kernel[(1,)](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )