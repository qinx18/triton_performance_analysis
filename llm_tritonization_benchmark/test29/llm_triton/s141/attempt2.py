import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for outer loop (i dimension)
    pid = tl.program_id(0)
    i = pid
    
    if i >= len_2d:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    i_plus_1 = i + 1
    k = i_plus_1 * (i_plus_1 - 1) // 2 + i_plus_1 - 1
    
    # Process inner loop sequentially since k has complex dependency
    for j in range(i, len_2d):
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * len_2d + i)
        
        # Load current flat_2d_array[k], add bb_val, store back
        current_val = tl.load(flat_2d_array_ptr + k)
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k for next iteration
        k += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    # Launch one thread per i value
    grid = (len_2d,)
    BLOCK_SIZE = 256
    
    s141_kernel[grid](bb, flat_2d_array, len_2d, BLOCK_SIZE)