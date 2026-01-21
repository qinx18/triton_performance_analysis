import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        aa_current_offsets = j * LEN_2D + i_idx
        aa_prev_offsets = (j - 1) * LEN_2D + i_idx
        bb_offsets = j * LEN_2D + i_idx
        
        aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=i_mask, other=0.0)
        bb_val = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
        
        result = aa_prev + bb_val
        
        tl.store(aa_ptr + aa_current_offsets, result, mask=i_mask)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    aa_flat = aa.view(-1)
    bb_flat = bb.view(-1)
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa_flat, bb_flat,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )