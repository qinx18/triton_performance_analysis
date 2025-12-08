import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, 
                 LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Vectorized j indices for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Sequential loop over i dimension (due to dependencies)
    for i in range(LEN_2D):
        # 2D array operations: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_indices = i * LEN_2D + j_idx
        bb_indices = i * LEN_2D + j_idx
        cc_indices = i * LEN_2D + j_idx
        
        # Load current values
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_indices, mask=j_mask, other=0.0)
        
        # Compute and store
        aa_new = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_indices, aa_new, mask=j_mask)
        
        # 1D array operation: a[i] = b[i] + c[i] * d[i]
        # Only first thread in first block handles this to avoid race conditions
        if pid == 0 and j_idx[0] == 0:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            result = b_val + c_val * d_val
            tl.store(a_ptr + i, result)

def s2275_triton(a, aa, b, bb, c, cc, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )