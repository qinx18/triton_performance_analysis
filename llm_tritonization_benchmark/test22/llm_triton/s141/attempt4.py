import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    # Simplifying: k = (i+1) * i / 2 + i = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_start = (i + 1) * i // 2 + i
    
    # Process j from i to LEN_2D-1 sequentially
    k = k_start
    for j in range(i, LEN_2D):
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        
        # Load flat_2d_array[k], add bb_val, and store back
        old_val = tl.load(flat_2d_array_ptr + k)
        tl.store(flat_2d_array_ptr + k, old_val + bb_val)
        
        # Update k for next iteration: k += j+1
        k += j + 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s141_kernel[grid](flat_2d_array, bb, LEN_2D, BLOCK_SIZE)