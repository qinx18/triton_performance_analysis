import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < len_2d * len_2d
    
    aa_vals = tl.load(aa_ptr + indices, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + indices, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + indices, mask=mask, other=0.0)
    
    result = aa_vals + bb_vals * cc_vals
    
    tl.store(flat_2d_array_ptr + indices, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array, len_2d):
    total_elements = len_2d * len_2d
    
    # Flatten 2D arrays for vectorized access
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    cc_flat = cc.flatten()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa_flat, bb_flat, cc_flat, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )