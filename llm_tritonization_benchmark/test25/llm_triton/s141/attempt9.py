import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= len_2d:
        return
    
    # Calculate starting k for this i
    k = (i + 1) * i // 2 + i
    
    # Process all j from i to len_2d-1
    for j in range(i, len_2d):
        # Create offset for bb[j][i]
        bb_offset = j * len_2d + i
        bb_val = tl.load(bb_ptr + bb_offset)
        
        # Create offset for flat_2d_array[k]
        old_val = tl.load(flat_2d_array_ptr + k)
        
        # Update flat_2d_array[k]
        new_val = old_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k for next iteration
        k += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    BLOCK_SIZE = 32
    grid = (len_2d,)
    
    s141_kernel[grid](
        bb, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )