import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D, i):
    # Calculate initial k for this i
    k = i * (i + 1) // 2 + i
    
    # Inner loop over j from i to LEN_2D-1
    for j in range(i, LEN_2D):
        # Load bb[j][i]
        bb_idx = j * LEN_2D + i
        bb_val = tl.load(bb_ptr + bb_idx)
        
        # Load flat_2d_array[k], add bb_val, store back
        current_val = tl.load(flat_2d_array_ptr + k)
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k for next iteration: k += j+1
        k += (j + 1)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch one kernel per i value
    for i in range(LEN_2D):
        grid = (1,)
        s141_kernel[grid](
            bb, flat_2d_array, LEN_2D, i
        )