import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one row i
    i = pid
    if i >= len_2d:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # This simplifies to: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_init = i * (i + 1) // 2 + i
    
    # Process inner loop sequentially since k depends on previous j values
    k_current = k_init
    for j in range(i, len_2d):
        # Load bb[j][i]
        bb_offset = j * len_2d + i
        bb_val = tl.load(bb_ptr + bb_offset)
        
        # Load current value from flat_2d_array[k]
        current_val = tl.load(flat_2d_array_ptr + k_current)
        
        # Update flat_2d_array[k]
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + k_current, new_val)
        
        # Update k for next iteration: k += j+1
        k_current += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    BLOCK_SIZE = 32
    grid = (len_2d,)
    
    s141_kernel[grid](
        bb, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )