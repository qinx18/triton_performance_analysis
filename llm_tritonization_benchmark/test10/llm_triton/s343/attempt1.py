import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Process elements sequentially since we need to maintain order
    k = 0
    
    # Process in blocks to handle large arrays efficiently
    for block_start in range(0, LEN_2D * LEN_2D, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, LEN_2D * LEN_2D)
        
        for idx in range(block_start, block_end):
            i = idx // LEN_2D
            j = idx % LEN_2D
            
            # Load bb[j][i] and aa[j][i]
            bb_idx = j * LEN_2D + i
            aa_idx = j * LEN_2D + i
            
            bb_val = tl.load(bb_ptr + bb_idx)
            
            if bb_val > 0.0:
                aa_val = tl.load(aa_ptr + aa_idx)
                tl.store(flat_2d_array_ptr + k, aa_val)
                k += 1

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Reset flat_2d_array to ensure clean state
    flat_2d_array.zero_()
    
    # Launch kernel with single thread block since we need sequential processing
    grid = (1,)
    
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return flat_2d_array