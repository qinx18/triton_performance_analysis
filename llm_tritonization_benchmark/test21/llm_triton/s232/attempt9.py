import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = (j_idx >= 1) & (j_idx < LEN_2D)
    
    for i in range(1, LEN_2D):
        # Only process j values where j >= i (triangular condition)
        valid_mask = j_mask & (j_idx >= i)
        
        # Load aa[j][i-1]
        prev_offsets = j_idx * LEN_2D + (i - 1)
        aa_prev = tl.load(aa_ptr + prev_offsets, mask=valid_mask, other=0.0)
        
        # Load bb[j][i]
        curr_offsets = j_idx * LEN_2D + i
        bb_curr = tl.load(bb_ptr + curr_offsets, mask=valid_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        result = aa_prev * aa_prev + bb_curr
        
        # Store result
        tl.store(aa_ptr + curr_offsets, result, mask=valid_mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s232_kernel[grid](aa, bb, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa