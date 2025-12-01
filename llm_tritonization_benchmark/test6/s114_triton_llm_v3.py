import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential implementation due to data dependencies
    for i in range(LEN_2D):
        # Process blocks of j values for current i
        for j_start in range(0, i, BLOCK_SIZE):
            j_offsets = tl.arange(0, BLOCK_SIZE)
            j_indices = j_start + j_offsets
            j_mask = (j_indices < i) & (j_indices >= 0)
            
            if tl.sum(j_mask.to(tl.int32)) > 0:
                # Calculate memory offsets for aa[i][j]
                aa_ij_offsets = i * LEN_2D + j_indices
                # Calculate memory offsets for aa[j][i] 
                aa_ji_offsets = j_indices * LEN_2D + i
                # Calculate memory offsets for bb[i][j]
                bb_ij_offsets = i * LEN_2D + j_indices
                
                # Load values
                aa_ji_vals = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask, other=0.0)
                bb_ij_vals = tl.load(bb_ptr + bb_ij_offsets, mask=j_mask, other=0.0)
                
                # Compute result
                result = aa_ji_vals + bb_ij_vals
                
                # Store result
                tl.store(aa_ptr + aa_ij_offsets, result, mask=j_mask)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with single program
    s114_kernel[(1,)](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa