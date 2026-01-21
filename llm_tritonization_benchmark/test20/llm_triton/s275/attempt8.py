import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa, bb, cc, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Load the first row to check condition
    aa_0_ptr = aa + i_idx
    aa_0_vals = tl.load(aa_0_ptr, mask=i_mask, other=0.0)
    
    # Check condition aa[0][i] > 0
    condition = aa_0_vals > 0.0
    combined_mask = i_mask & condition
    
    # Sequential loop over j dimension
    for j in range(1, N):
        # Load aa[j-1][i] (previous row)
        aa_prev_ptr = aa + (j - 1) * N + i_idx
        aa_prev_vals = tl.load(aa_prev_ptr, mask=combined_mask, other=0.0)
        
        # Load bb[j][i] and cc[j][i]
        bb_ptr = bb + j * N + i_idx
        cc_ptr = cc + j * N + i_idx
        bb_vals = tl.load(bb_ptr, mask=combined_mask, other=0.0)
        cc_vals = tl.load(cc_ptr, mask=combined_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        new_vals = aa_prev_vals + bb_vals * cc_vals
        
        # Store back to aa[j][i]
        aa_curr_ptr = aa + j * N + i_idx
        tl.store(aa_curr_ptr, new_vals, mask=combined_mask)

def s275_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Ensure arrays are contiguous
    aa = aa.contiguous()
    bb = bb.contiguous() 
    cc = cc.contiguous()
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s275_kernel[grid](aa, bb, cc, N, BLOCK_SIZE=BLOCK_SIZE)