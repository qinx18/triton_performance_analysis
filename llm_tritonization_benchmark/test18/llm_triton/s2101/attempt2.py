import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, n_dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_dim
    
    # Calculate diagonal indices: [i, i] -> i * n_dim + i
    diag_indices = indices * n_dim + indices
    
    # Load diagonal elements
    aa_vals = tl.load(aa_ptr + diag_indices, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + diag_indices, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + diag_indices, mask=mask, other=0.0)
    
    # Compute: aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(aa_ptr + diag_indices, result, mask=mask)

def s2101_triton(aa, bb, cc):
    n_dim = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_dim, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc,
        n_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa