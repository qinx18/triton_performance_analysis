import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, LEN_2D: tl.constexpr):
    # Each program handles one i value
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k
    k = (i + 1) * (i + 1 - 1) // 2 + (i + 1) - 1
    
    # Process j loop sequentially
    for j in range(i, LEN_2D):
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        
        # Load flat_2d_array[k]
        old_val = tl.load(flat_2d_array_ptr + k)
        
        # Update flat_2d_array[k]
        new_val = old_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k for next iteration
        k += j + 1

def s141_triton(flat_2d_array, bb):
    LEN_2D = bb.shape[0]
    
    grid = (LEN_2D,)
    s141_kernel[grid](
        flat_2d_array,
        bb,
        LEN_2D
    )
    
    return flat_2d_array