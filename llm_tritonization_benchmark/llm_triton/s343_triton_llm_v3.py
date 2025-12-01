import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(
    aa_ptr, bb_ptr, flat_2d_array_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # This kernel implements stream compaction (packing)
    # We need to process sequentially to maintain order
    
    k = -1
    
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            # Load bb[j][i] and aa[j][i]
            bb_idx = j * LEN_2D + i
            bb_val = tl.load(bb_ptr + bb_idx)
            
            if bb_val > 0.0:
                k += 1
                aa_val = tl.load(aa_ptr + bb_idx)
                tl.store(flat_2d_array_ptr + k, aa_val)

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    
    # Ensure arrays are contiguous
    aa = aa.contiguous()
    bb = bb.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    # Launch single thread since this is inherently sequential
    grid = (1,)
    
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=256
    )
    
    return flat_2d_array