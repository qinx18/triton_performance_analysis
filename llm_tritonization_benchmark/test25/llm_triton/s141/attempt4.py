import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, i):
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    k_start = i * (i + 1) // 2 + i
    current_k = k_start
    
    # Process j values from i to len_2d-1 sequentially
    for j in range(i, len_2d):
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
    for i in range(N):
        s141_kernel[(1,)](
            bb, flat_2d_array, N, i
        )
    
    return flat_2d_array