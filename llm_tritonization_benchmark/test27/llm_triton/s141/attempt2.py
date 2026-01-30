import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= len_2d:
        return
    
    # Calculate initial k: (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # Simplifying: (i+1) * i / 2 + i = i * (i + 1) / 2 + i
    k = i * (i + 1) // 2 + i
    
    # Process inner loop for this i
    for j in range(i, len_2d):
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * len_2d + i)
        
        # Load flat_2d_array[k]
        old_val = tl.load(flat_2d_array_ptr + k)
        
        # Update flat_2d_array[k]
        new_val = old_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k for next iteration: k += j + 1
        k += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    BLOCK_SIZE = 128
    grid = (len_2d,)
    
    s141_kernel[grid](
        bb, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )