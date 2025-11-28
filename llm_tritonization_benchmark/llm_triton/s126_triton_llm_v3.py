import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, flat_2d_array_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    k = i * LEN_2D + 1
    
    for j in range(1, LEN_2D):
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * LEN_2D + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + k - 1)
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute and store bb[j][i]
        bb_offset = j * LEN_2D + i
        result = bb_prev + flat_val * cc_val
        tl.store(bb_ptr + bb_offset, result)
        
        k += 1
    
    k += 1

def s126_triton(bb, flat_2d_array, cc):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with i-parallel strategy
    grid = (LEN_2D,)
    BLOCK_SIZE = 1
    
    s126_kernel[grid](
        bb, flat_2d_array, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )