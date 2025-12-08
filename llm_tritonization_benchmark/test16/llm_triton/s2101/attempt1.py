import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Calculate 2D diagonal indices: [i, i] -> i * n + i where n = sqrt(n_elements)
    n_dim = tl.sqrt(n_elements.to(tl.float32)).to(tl.int32)
    diag_offsets = indices * n_dim + indices
    diag_mask = mask & (indices < n_dim)
    
    # Load diagonal elements
    bb_vals = tl.load(bb_ptr + diag_offsets, mask=diag_mask, other=0.0)
    cc_vals = tl.load(cc_ptr + diag_offsets, mask=diag_mask, other=0.0)
    aa_vals = tl.load(aa_ptr + diag_offsets, mask=diag_mask, other=0.0)
    
    # Compute: aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(aa_ptr + diag_offsets, result, mask=diag_mask)

def s2101_triton(aa, bb, cc):
    n_dim = aa.shape[0]
    n_elements = n_dim
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa