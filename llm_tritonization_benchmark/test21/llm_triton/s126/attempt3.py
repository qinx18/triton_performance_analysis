import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Process column i sequentially for j dimension (due to recurrence)
    for j in range(1, LEN_2D):
        # Calculate k index based on position in the nested loops
        # k starts at 1, increments for each (i,j) pair, plus extra increment after each i
        k = 1 + i * (LEN_2D - 1) + i + (j - 1)
        
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * LEN_2D + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + (k - 1))
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute and store bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev + flat_val * cc_val
        bb_offset = j * LEN_2D + i
        tl.store(bb_ptr + bb_offset, result)

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, 1),)
    s126_kernel[grid](bb, cc, flat_2d_array, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)