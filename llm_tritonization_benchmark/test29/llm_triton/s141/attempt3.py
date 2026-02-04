import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one value of i
    i = pid
    
    if i >= len_2d:
        return
    
    # Compute initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # Simplifying: k = i * (i+1) / 2 + i = i * (i+3) / 2
    k = i * (i + 1) // 2 + i
    
    # Process j loop sequentially
    for j in range(i, len_2d):
        # Load current flat_2d_array[k]
        current_val = tl.load(flat_2d_array_ptr + k)
        
        # Add bb[j][i] to flat_2d_array[k]
        bb_val = tl.load(bb_ptr + j * len_2d + i)
        new_val = current_val + bb_val
        
        # Store back
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k for next iteration: k += j+1
        k += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    # Launch one thread per i value
    grid = (len_2d,)
    
    s141_kernel[grid](
        bb,
        flat_2d_array,
        len_2d,
        BLOCK_SIZE=32
    )