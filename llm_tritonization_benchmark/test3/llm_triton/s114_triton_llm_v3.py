import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential implementation due to data dependencies
    # Each program handles one row to avoid race conditions
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Process all j < i for this row i
    for j in range(i):
        # Calculate memory offsets
        aa_ij_offset = i * LEN_2D + j
        aa_ji_offset = j * LEN_2D + i
        bb_ij_offset = i * LEN_2D + j
        
        # Load values
        aa_ji_val = tl.load(aa_ptr + aa_ji_offset)
        bb_ij_val = tl.load(bb_ptr + bb_ij_offset)
        
        # Compute and store
        result = aa_ji_val + bb_ij_val
        tl.store(aa_ptr + aa_ij_offset, result)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 1
    
    # Launch kernel with one program per row
    grid = (LEN_2D,)
    
    s114_kernel[grid](
        aa, bb, LEN_2D, BLOCK_SIZE
    )
    
    return aa