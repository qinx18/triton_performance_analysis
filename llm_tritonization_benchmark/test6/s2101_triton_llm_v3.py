import triton
import triton.language as tl
import torch

@triton.jit
def s2101_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Calculate diagonal indices (i*n_elements + i for aa[i][i])
    diag_offsets = indices * n_elements + indices
    
    # Load diagonal elements
    aa_vals = tl.load(aa_ptr + diag_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + diag_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + diag_offsets, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(aa_ptr + diag_offsets, result, mask=mask)

def s2101_triton(aa, bb, cc):
    n_elements = aa.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return aa