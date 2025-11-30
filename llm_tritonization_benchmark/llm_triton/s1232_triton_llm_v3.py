import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the j dimension
    j = tl.program_id(0)
    
    if j >= LEN_2D:
        return
    
    # Process elements in blocks along the i dimension
    i_offset = tl.program_id(1) * BLOCK_SIZE
    i_range = tl.arange(0, BLOCK_SIZE)
    i = i_offset + i_range
    
    # Mask for valid i indices and triangular constraint (i >= j)
    mask = (i < LEN_2D) & (i >= j)
    
    # Calculate memory offsets for 2D arrays (row-major order)
    offsets = i * LEN_2D + j
    
    # Load data with masking
    bb_vals = tl.load(bb_ptr + offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + offsets, mask=mask)
    
    # Perform computation
    result = bb_vals + cc_vals
    
    # Store result with masking
    tl.store(aa_ptr + offsets, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Calculate grid dimensions
    # Grid dimension 0: j values (0 to LEN_2D-1)
    # Grid dimension 1: blocks of i values
    grid_j = LEN_2D
    grid_i = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    s1232_kernel[(grid_j, grid_i)](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )