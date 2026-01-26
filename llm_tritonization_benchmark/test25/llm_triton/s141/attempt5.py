import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get the program ID for the i dimension
    pid = tl.program_id(0)
    i = pid
    
    if i >= len_2d:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    k_start = i * (i + 1) // 2 + i
    
    # Process j values from i to len_2d-1 in blocks
    j_start = i
    j_end = len_2d
    current_k = k_start
    
    for j in range(j_start, j_end):
        # Load bb[j][i]
        bb_idx = j * len_2d + i
        bb_val = tl.load(bb_ptr + bb_idx)
        
        # Load current value from flat_2d_array
        current_val = tl.load(flat_2d_array_ptr + current_k)
        
        # Update flat_2d_array[k]
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + current_k, new_val)
        
        # Update k for next iteration: k += j+1
        current_k += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    # Get dimensions
    N = len_2d
    
    # Launch kernel with one thread per i value
    BLOCK_SIZE = 1
    grid = (N,)
    
    s141_kernel[grid](
        bb, flat_2d_array, N, BLOCK_SIZE
    )
    
    return flat_2d_array