import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid_i = tl.program_id(0)
    
    if pid_i >= LEN_2D:
        return
    
    # For each i, process all j values sequentially due to recurrence
    k = 1 + pid_i * LEN_2D
    for j in range(1, LEN_2D):
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * LEN_2D + pid_i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_offset = k - 1
        flat_val = tl.load(flat_2d_array_ptr + flat_offset)
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + pid_i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev + flat_val * cc_val
        
        # Store bb[j][i]
        bb_offset = j * LEN_2D + pid_i
        tl.store(bb_ptr + bb_offset, result)
        
        k += 1
    
    # Extra increment of k after inner loop (++k after j loop)

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, 1),)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )