import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(len_2d):
        # Load aa[i][j]
        aa_ptrs = aa_ptr + i * len_2d + j_idx
        aa_vals = tl.load(aa_ptrs, mask=j_mask)
        
        # Load cc[j][i] - note the transposed indexing
        cc_ptrs = cc_ptr + j_idx * len_2d + i
        cc_vals = tl.load(cc_ptrs, mask=j_mask)
        
        # Load bb[i][j]
        bb_ptrs = bb_ptr + i * len_2d + j_idx
        bb_vals = tl.load(bb_ptrs, mask=j_mask)
        
        # Compute aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result back to aa[i][j]
        tl.store(aa_ptrs, result, mask=j_mask)

def s1115_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )