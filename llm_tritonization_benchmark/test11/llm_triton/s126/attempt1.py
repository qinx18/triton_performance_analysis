import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles one column i
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate starting k for this column
    k = 1 + i * LEN_2D
    
    # Process column i sequentially (j loop cannot be parallelized due to recurrence)
    for j in range(1, LEN_2D):
        # Load bb[j-1][i] (previous row, same column)
        prev_bb = tl.load(bb_ptr + (j-1) * LEN_2D + i)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + k - 1)
        
        # Load cc[j][i]
        cc_val = tl.load(cc_ptr + j * LEN_2D + i)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = prev_bb + flat_val * cc_val
        
        # Store bb[j][i]
        tl.store(bb_ptr + j * LEN_2D + i, result)
        
        k += 1
    
def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Grid: one program per column
    grid = (LEN_2D,)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=1
    )