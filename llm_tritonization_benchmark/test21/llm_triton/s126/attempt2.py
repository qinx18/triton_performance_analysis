import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Process column i with blocked approach for j dimension
    for j_start in range(1, LEN_2D, BLOCK_SIZE_J):
        j_end = min(j_start + BLOCK_SIZE_J, LEN_2D)
        
        for j in range(j_start, j_end):
            # Calculate k index: k starts at 1, for each (i,j) pair
            # k = 1 + i * LEN_2D + (j - 1)
            # But there's an extra ++k at end of each i loop
            # So k = 1 + i * (LEN_2D - 1) + i + (j - 1) = 1 + i * LEN_2D + (j - 1)
            k = 1 + i * LEN_2D + (j - 1)
            
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
    
    # Process each column (i) in a separate program
    for i in range(LEN_2D):
        k_base = 1 + i * LEN_2D
        for j in range(1, LEN_2D):
            k = k_base + (j - 1)
            bb[j, i] = bb[j-1, i] + flat_2d_array[k-1] * cc[j, i]