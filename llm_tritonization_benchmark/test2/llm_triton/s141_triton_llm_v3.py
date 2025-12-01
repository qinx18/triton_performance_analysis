import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k = i * (i + 1) // 2 + i
    
    # Process the inner j loop sequentially
    for j in range(i, LEN_2D):
        # Load bb[j][i]
        bb_offset = j * LEN_2D + i
        bb_val = tl.load(bb_ptr + bb_offset)
        
        # Load current value from flat_2d_array[k]
        current_val = tl.load(flat_2d_array_ptr + k)
        
        # Update flat_2d_array[k]
        tl.store(flat_2d_array_ptr + k, current_val + bb_val)
        
        # Update k for next iteration: k += j+1
        k += j + 1

def s141_triton(flat_2d_array, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, 1),)
    
    s141_kernel[grid](
        flat_2d_array,
        bb,
        LEN_2D,
        BLOCK_SIZE
    )
    
    return flat_2d_array