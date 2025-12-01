import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, flat_2d_array_ptr, cc_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    mask = offsets < LEN_2D
    
    # Calculate k for this i_val and j values
    k_vals = i_val * LEN_2D + offsets
    
    # Load bb[j-1][i] values
    bb_prev_ptrs = bb_ptr + (offsets - 1) * LEN_2D + i_val
    bb_prev_vals = tl.load(bb_prev_ptrs, mask=mask)
    
    # Load flat_2d_array[k-1] values
    flat_ptrs = flat_2d_array_ptr + k_vals - 1
    flat_vals = tl.load(flat_ptrs, mask=mask)
    
    # Load cc[j][i] values
    cc_ptrs = cc_ptr + offsets * LEN_2D + i_val
    cc_vals = tl.load(cc_ptrs, mask=mask)
    
    # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
    result = bb_prev_vals + flat_vals * cc_vals
    
    # Store result
    bb_ptrs = bb_ptr + offsets * LEN_2D + i_val
    tl.store(bb_ptrs, result, mask=mask)

def s126_triton(bb, flat_2d_array, cc):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s126_kernel[grid](bb, flat_2d_array, cc, i, LEN_2D, BLOCK_SIZE)