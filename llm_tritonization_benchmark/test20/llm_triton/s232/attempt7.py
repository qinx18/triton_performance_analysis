import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_indices < LEN_2D
    
    for i in range(1, LEN_2D):
        # Current i mask: only process j values where i <= j
        i_mask = i <= j_indices
        combined_mask = j_mask & i_mask
        
        # Calculate offsets for aa[j][i-1], aa[j][i], and bb[j][i]
        aa_prev_offsets = j_indices * LEN_2D + (i - 1)
        aa_curr_offsets = j_indices * LEN_2D + i
        bb_offsets = j_indices * LEN_2D + i
        
        # Load aa[j][i-1] and bb[j][i]
        aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=combined_mask, other=0.0)
        bb_val = tl.load(bb_ptr + bb_offsets, mask=combined_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        result = aa_prev * aa_prev + bb_val
        
        # Store result back to aa[j][i]
        tl.store(aa_ptr + aa_curr_offsets, result, mask=combined_mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    num_j_blocks = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)
    grid = (num_j_blocks,)
    
    s232_kernel[grid](aa, bb, LEN_2D, BLOCK_SIZE)