import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    # Check condition: aa[0][i] > 0
    aa_0_ptr = aa_ptr + i_idx  # aa[0][i]
    aa_0_vals = tl.load(aa_0_ptr, mask=i_mask, other=0.0)
    condition_mask = aa_0_vals > 0.0
    active_mask = i_mask & condition_mask
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        if tl.sum(active_mask.to(tl.int32)) == 0:
            break
            
        # Load aa[j-1][i]
        aa_prev_ptr = aa_ptr + (j - 1) * LEN_2D + i_idx
        aa_prev_vals = tl.load(aa_prev_ptr, mask=active_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptr_curr = bb_ptr + j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptr_curr, mask=active_mask, other=0.0)
        
        # Load cc[j][i]
        cc_ptr_curr = cc_ptr + j * LEN_2D + i_idx
        cc_vals = tl.load(cc_ptr_curr, mask=active_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_prev_vals + bb_vals * cc_vals
        
        # Store aa[j][i]
        aa_curr_ptr = aa_ptr + j * LEN_2D + i_idx
        tl.store(aa_curr_ptr, result, mask=active_mask)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )