import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for parallelizing over columns
    pid = tl.program_id(0)
    
    # Each program handles one column i
    i = pid
    if i >= len_2d:
        return
    
    # Process column i sequentially due to recurrence dependency
    for j in range(1, len_2d):
        # Calculate k index: k starts at 1, increments for each (i,j) pair
        # k = 1 + i * len_2d + (j - 1)
        k = 1 + i * len_2d + (j - 1)
        
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * len_2d + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_offset = k - 1
        flat_val = tl.load(flat_2d_array_ptr + flat_offset)
        
        # Load cc[j][i]
        cc_offset = j * len_2d + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev + flat_val * cc_val
        
        # Store bb[j][i]
        bb_offset = j * len_2d + i
        tl.store(bb_ptr + bb_offset, result)

def s126_triton(bb, cc, flat_2d_array, len_2d):
    # Launch kernel with one program per column
    grid = (len_2d,)
    BLOCK_SIZE = 1
    
    s126_kernel[grid](
        bb, cc, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )