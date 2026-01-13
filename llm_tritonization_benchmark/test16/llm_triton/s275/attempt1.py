import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, stride_aa_0, stride_aa_1, stride_bb_0, stride_bb_1, stride_cc_0, stride_cc_1, n_cols, n_rows, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < n_cols
    
    # Check condition aa[0][i] > 0
    aa_0_ptr = aa_ptr + i_idx * stride_aa_1
    aa_0_vals = tl.load(aa_0_ptr, mask=mask_i, other=0.0)
    condition = aa_0_vals > 0.0
    
    # Combined mask for valid indices and condition
    active_mask = mask_i & condition
    
    if tl.sum(active_mask.to(tl.int32)) > 0:
        # Initialize with aa[0][i] values
        prev_vals = aa_0_vals
        
        for j in range(1, n_rows):
            # Load bb[j][i] and cc[j][i]
            bb_ptr_j = bb_ptr + j * stride_bb_0 + i_idx * stride_bb_1
            cc_ptr_j = cc_ptr + j * stride_cc_0 + i_idx * stride_cc_1
            
            bb_vals = tl.load(bb_ptr_j, mask=active_mask, other=0.0)
            cc_vals = tl.load(cc_ptr_j, mask=active_mask, other=0.0)
            
            # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
            new_vals = prev_vals + bb_vals * cc_vals
            
            # Store result
            aa_ptr_j = aa_ptr + j * stride_aa_0 + i_idx * stride_aa_1
            tl.store(aa_ptr_j, new_vals, mask=active_mask)
            
            # Update previous values for next iteration
            prev_vals = new_vals

def s275_triton(aa, bb, cc):
    n_rows, n_cols = aa.shape
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_cols, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        aa.stride(0), aa.stride(1),
        bb.stride(0), bb.stride(1),
        cc.stride(0), cc.stride(1),
        n_cols, n_rows,
        BLOCK_SIZE=BLOCK_SIZE
    )