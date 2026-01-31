import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < len_2d
    
    # Calculate diagonal positions: aa[i][i] means row i, column i
    diagonal_offsets = indices * len_2d + indices
    
    # Load diagonal elements
    aa_vals = tl.load(aa_ptr + diagonal_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + diagonal_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + diagonal_offsets, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store back to diagonal elements
    tl.store(aa_ptr + diagonal_offsets, result, mask=mask)

def s2101_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )