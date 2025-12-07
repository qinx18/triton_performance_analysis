import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, n_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < n_size
    
    for i in range(n_size):
        # Load aa[i][j] values
        aa_ptrs = aa_ptr + i * n_size + j_idx
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        
        # Load cc[j][i] values (note the transposed indexing)
        cc_ptrs = cc_ptr + j_idx * n_size + i
        cc_vals = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        
        # Load bb[i][j] values
        bb_ptrs = bb_ptr + i * n_size + j_idx
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store back to aa[i][j]
        tl.store(aa_ptrs, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    n_size = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_size, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc, n_size, BLOCK_SIZE
    )