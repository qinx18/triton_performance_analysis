import triton
import triton.language as tl
import torch

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Check condition aa[0][i] > 0
    condition_ptr = aa_ptr + i_idx
    condition_vals = tl.load(condition_ptr, mask=i_mask, other=0.0)
    condition_mask = condition_vals > 0.0
    
    # Combined mask for valid indices and condition
    valid_mask = i_mask & condition_mask
    
    # Sequential loop over j dimension
    for j in range(1, N):
        # Load aa[j-1][i]
        prev_aa_ptr = aa_ptr + (j - 1) * N + i_idx
        prev_aa_vals = tl.load(prev_aa_ptr, mask=i_mask, other=0.0)
        
        # Load bb[j][i] and cc[j][i]
        bb_ptr_curr = bb_ptr + j * N + i_idx
        cc_ptr_curr = cc_ptr + j * N + i_idx
        bb_vals = tl.load(bb_ptr_curr, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr_curr, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = prev_aa_vals + bb_vals * cc_vals
        
        # Store the result only where condition is true
        aa_ptr_curr = aa_ptr + j * N + i_idx
        tl.store(aa_ptr_curr, result, mask=valid_mask)

def s275_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )