import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    # Check condition aa[0][i] > 0
    condition_ptrs = aa_ptr + i_idx
    condition_vals = tl.load(condition_ptrs, mask=i_mask, other=0.0)
    condition_mask = condition_vals > 0.0
    active_mask = i_mask & condition_mask
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] for active columns
        prev_ptrs = aa_ptr + (j - 1) * LEN_2D + i_idx
        prev_vals = tl.load(prev_ptrs, mask=active_mask, other=0.0)
        
        # Load bb[j][i] for active columns
        bb_ptrs = bb_ptr + j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptrs, mask=active_mask, other=0.0)
        
        # Load cc[j][i] for active columns
        cc_ptrs = cc_ptr + j * LEN_2D + i_idx
        cc_vals = tl.load(cc_ptrs, mask=active_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = prev_vals + bb_vals * cc_vals
        
        # Store result only where condition is met and indices are valid
        aa_ptrs = aa_ptr + j * LEN_2D + i_idx
        tl.store(aa_ptrs, result, mask=active_mask)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s275_kernel[grid](aa, bb, cc, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)