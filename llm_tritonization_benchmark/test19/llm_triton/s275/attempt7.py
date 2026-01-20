import triton
import triton.language as tl
import torch

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Load aa[0][i] for condition check
    aa_0_vals = tl.load(aa_ptr + i_idx, mask=i_mask, other=0.0)
    condition_mask = (aa_0_vals > 0.0) & i_mask
    
    # Sequential loop over j dimension
    for j in range(1, N):
        # Load aa[j-1][i]
        aa_prev_ptr = aa_ptr + (j-1) * N + i_idx
        aa_prev_vals = tl.load(aa_prev_ptr, mask=condition_mask, other=0.0)
        
        # Load bb[j][i] and cc[j][i]
        bb_ptr_offset = bb_ptr + j * N + i_idx
        cc_ptr_offset = cc_ptr + j * N + i_idx
        bb_vals = tl.load(bb_ptr_offset, mask=condition_mask, other=0.0)
        cc_vals = tl.load(cc_ptr_offset, mask=condition_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_prev_vals + bb_vals * cc_vals
        
        # Store result
        aa_curr_ptr = aa_ptr + j * N + i_idx
        tl.store(aa_curr_ptr, result, mask=condition_mask)

def s275_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s275_kernel[grid](aa, bb, cc, N, BLOCK_SIZE=BLOCK_SIZE)