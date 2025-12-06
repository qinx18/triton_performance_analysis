import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(
    bb_ptr,
    flat_2d_array_ptr,
    LEN_2D: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k value: (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    k = (i + 1) * i // 2 + i
    
    # Sequential loop over j from i to LEN_2D-1
    for j_offset in range(LEN_2D):
        j = i + j_offset
        if j >= LEN_2D:
            return
            
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        
        # Load current flat_2d_array[k]
        current_val = tl.load(flat_2d_array_ptr + k)
        
        # Update flat_2d_array[k]
        tl.store(flat_2d_array_ptr + k, current_val + bb_val)
        
        # Update k: k += j+1
        k = k + j + 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with one thread per i
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb,
        flat_2d_array,
        LEN_2D=LEN_2D,
    )
    
    return flat_2d_array