import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load bb[i][i] and cc[i][i] values
    bb_vals = tl.load(bb_ptr + indices * n_elements + indices, mask=mask)
    cc_vals = tl.load(cc_ptr + indices * n_elements + indices, mask=mask)
    
    # Load current aa[i][i] values
    aa_vals = tl.load(aa_ptr + indices * n_elements + indices, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result back to aa[i][i]
    tl.store(aa_ptr + indices * n_elements + indices, result, mask=mask)

def s2101_triton(aa, bb, cc):
    n_elements = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc, n_elements, BLOCK_SIZE
    )