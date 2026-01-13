import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, n):
    BLOCK_SIZE: tl.constexpr = 256
    
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < n
    
    for i in range(n):
        # Calculate memory addresses for row i
        aa_row_ptr = aa_ptr + i * n
        bb_row_ptr = bb_ptr + i * n
        
        # Load aa[i][j] and bb[i][j]
        aa_vals = tl.load(aa_row_ptr + j_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_row_ptr + j_idx, mask=j_mask, other=0.0)
        
        # Load cc[j][i] - note the transposed indices
        cc_col_ptr = cc_ptr + i
        cc_offsets = j_idx * n + i
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
        
        # Compute: aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result back to aa[i][j]
        tl.store(aa_row_ptr + j_idx, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc, n
    )
    
    return aa