import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        # Current row indices for aa[j][i] and bb[j][i]
        curr_aa_idx = j * LEN_2D + i_idx
        curr_bb_idx = j * LEN_2D + i_idx
        
        # Previous row indices for aa[j-1][i]
        prev_aa_idx = (j - 1) * LEN_2D + i_idx
        
        # Load aa[j-1][i] and bb[j][i]
        prev_aa = tl.load(aa_ptr + prev_aa_idx, mask=i_mask)
        bb_val = tl.load(bb_ptr + curr_bb_idx, mask=i_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = prev_aa + bb_val
        
        # Store result
        tl.store(aa_ptr + curr_aa_idx, result, mask=i_mask)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )