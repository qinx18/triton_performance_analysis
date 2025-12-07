import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k: (i+1) * ((i+1) - 1) / 2 + (i+1) - 1 = i * (i+1) / 2 + i
    k = i * (i + 1) // 2 + i
    
    # Process j loop sequentially
    j = i
    while j < LEN_2D:
        # Load bb[j][i]
        bb_idx = j * LEN_2D + i
        bb_val = tl.load(bb_ptr + bb_idx)
        
        # Load current flat_2d_array[k], add bb_val, store back
        current_val = tl.load(flat_2d_array_ptr + k)
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k += j + 1
        k += j + 1
        j += 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with one thread per outer i
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb, flat_2d_array, LEN_2D
    )
    
    return flat_2d_array