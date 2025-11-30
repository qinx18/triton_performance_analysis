import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get thread ID
    pid = tl.program_id(0)
    
    # Each thread handles one i value
    i = pid
    
    # Bounds check for i
    if i >= LEN_2D:
        return
    
    # Sequential loop over j from 0 to i-1
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
    BLOCK_SIZE = 256
    
    # Calculate grid size - one thread per i value
    grid = (triton.cdiv(LEN_2D, 1),)
    
    # Launch kernel
    s114_kernel[grid](
        aa, bb, LEN_2D, BLOCK_SIZE
    )
    
    return aa