import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(
    bb_ptr,
    cc_ptr,
    flat_2d_array_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Initialize k for this column
    k = 1 + i * LEN_2D
    
    # Sequential processing for j dimension due to data dependency
    for j in range(1, LEN_2D):
        # Load bb[j-1][i]
        bb_prev_idx = (j - 1) * LEN_2D + i
        bb_prev = tl.load(bb_ptr + bb_prev_idx)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + (k - 1))
        
        # Load cc[j][i]
        cc_idx = j * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_idx)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev + flat_val * cc_val
        
        # Store bb[j][i]
        bb_idx = j * LEN_2D + i
        tl.store(bb_ptr + bb_idx, result)
        
        k += 1

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s126_kernel[grid](
        bb,
        cc,
        flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )