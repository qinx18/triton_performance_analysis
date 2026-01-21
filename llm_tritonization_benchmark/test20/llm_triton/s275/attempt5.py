import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get column block
    pid = tl.program_id(0)
    
    # Column offsets
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Check condition aa[0][i] > 0
    aa_0_ptr = aa_ptr + i_idx
    aa_0_vals = tl.load(aa_0_ptr, mask=i_mask, other=0.0)
    condition_mask = aa_0_vals > 0.0
    
    # Combined mask for valid indices and condition
    active_mask = i_mask & condition_mask
    
    # Sequential loop over j (rows) starting from j=1
    for j in range(1, N):
        # Load aa[j-1][i] (previous row)
        prev_j = j - 1
        aa_prev_ptr = aa_ptr + prev_j * N + i_idx
        aa_prev_vals = tl.load(aa_prev_ptr, mask=i_mask, other=0.0)
        
        # Load bb[j][i] and cc[j][i] (current row)
        bb_ptr_curr = bb_ptr + j * N + i_idx
        cc_ptr_curr = cc_ptr + j * N + i_idx
        bb_vals = tl.load(bb_ptr_curr, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr_curr, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_prev_vals + bb_vals * cc_vals
        
        # Store result back to aa[j][i] only where condition is met
        result = tl.where(condition_mask, result, tl.load(aa_ptr + j * N + i_idx, mask=i_mask, other=0.0))
        
        aa_curr_ptr = aa_ptr + j * N + i_idx
        tl.store(aa_curr_ptr, result, mask=i_mask)

def s275_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa