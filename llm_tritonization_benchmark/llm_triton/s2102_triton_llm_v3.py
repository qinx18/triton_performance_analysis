import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the outer loop (i dimension)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Zero out column i: aa[j][i] = 0 for all j
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Process in blocks for the j dimension
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_valid = j_indices < LEN_2D
        mask = j_mask & j_valid
        
        # Calculate memory addresses: aa[j][i] = aa + j*LEN_2D + i
        mem_offsets = j_indices * LEN_2D + i
        
        # Store zeros
        tl.store(aa_ptr + mem_offsets, 0.0, mask=mask)
    
    # Set diagonal element: aa[i][i] = 1
    diagonal_offset = i * LEN_2D + i
    tl.store(aa_ptr + diagonal_offset, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(min(1024, LEN_2D))
    
    # Launch kernel with one thread per row (i dimension)
    grid = (LEN_2D,)
    
    s2102_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )