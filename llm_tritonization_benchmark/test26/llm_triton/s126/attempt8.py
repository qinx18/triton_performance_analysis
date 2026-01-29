import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr):
    # Each program handles one column i
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Initialize k for this column: k starts at 1, then for each column we advance by LEN_2D
    k = 1 + i * LEN_2D
    
    # Sequential loop over rows j (j=1 to LEN_2D-1)
    for j in range(1, LEN_2D):
        # Load bb[j-1][i] (previous row, same column)
        bb_prev_offset = (j - 1) * LEN_2D + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1] (k starts from 1, so k-1 is the 0-based index)
        flat_offset = k - 1
        flat_val = tl.load(flat_2d_array_ptr + flat_offset)
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        bb_new = bb_prev + flat_val * cc_val
        
        # Store bb[j][i]
        bb_offset = j * LEN_2D + i
        tl.store(bb_ptr + bb_offset, bb_new)
        
        # Increment k
        k = k + 1

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with one program per column
    grid = (LEN_2D,)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D
    )