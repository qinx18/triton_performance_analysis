import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which (i, j) pair this program handles
    total_elements = (LEN_2D * (LEN_2D + 1)) // 2
    
    if pid >= total_elements:
        return
    
    # Convert linear index to triangular (i, j) coordinates
    # For triangular iteration where i >= j
    j = 0
    cumulative = 0
    remaining = pid
    
    # Find the j value
    for k in range(LEN_2D):
        elements_in_col = LEN_2D - k
        if remaining < elements_in_col:
            j = k
            i = j + remaining
            break
        remaining -= elements_in_col
    
    # Bounds check
    if i >= LEN_2D or j >= LEN_2D or i < j:
        return
    
    # Calculate flat index for 2D arrays
    idx = i * LEN_2D + j
    
    # Load values
    bb_val = tl.load(bb_ptr + idx)
    cc_val = tl.load(cc_ptr + idx)
    
    # Compute and store
    result = bb_val + cc_val
    tl.store(aa_ptr + idx, result)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Total number of elements in upper triangular matrix (including diagonal)
    total_elements = (LEN_2D * (LEN_2D + 1)) // 2
    
    BLOCK_SIZE = 256
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s1232_kernel[grid_size](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )