import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Process one column (i) per program
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Sequential processing of j from 1 to LEN_2D-1
    k = i * LEN_2D + 1  # Starting k value for this column
    
    for j in range(1, LEN_2D):
        # Load bb[j-1][i] 
        bb_prev_offset = (j-1) * LEN_2D + i
        bb_prev_val = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + k - 1)
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev_val + flat_val * cc_val
        
        # Store result
        bb_offset = j * LEN_2D + i
        tl.store(bb_ptr + bb_offset, result)
        
        k += 1
    
    # Final increment of k (matches the ++k after inner loop in C)
    # This doesn't affect computation but maintains k consistency

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with one thread per column
    BLOCK_SIZE = 1  # Not used in this kernel but required for compilation
    grid = (LEN_2D,)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )