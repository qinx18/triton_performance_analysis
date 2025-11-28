import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get thread ID
    pid = tl.program_id(0)
    
    # Calculate which (i, j) pair this thread handles
    total_elements = (LEN_2D * (LEN_2D + 1)) // 2  # triangular matrix elements
    
    if pid >= total_elements:
        return
    
    # Convert linear index to (i, j) coordinates for triangular access
    # For triangular matrix where i >= j, we need to find j first
    j = 0
    remaining = pid
    while remaining >= (LEN_2D - j):
        remaining -= (LEN_2D - j)
        j += 1
    
    i = j + remaining
    
    # Bounds check
    if i >= LEN_2D or j >= LEN_2D or i < j:
        return
    
    # Calculate linear index for 2D arrays stored in row-major order
    idx = i * LEN_2D + j
    
    # Load values
    bb_val = tl.load(bb_ptr + idx)
    cc_val = tl.load(cc_ptr + idx)
    
    # Compute and store
    result = bb_val + cc_val
    tl.store(aa_ptr + idx, result)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Calculate total number of elements in triangular matrix
    total_elements = (LEN_2D * (LEN_2D + 1)) // 2
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )