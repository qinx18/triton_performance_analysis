import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ((LEN_2D - 1) * LEN_2D) // 2
    
    # Convert linear index to j, i coordinates for triangular iteration
    # We need to map linear indices to (j, i) where j in [1, LEN_2D) and i in [1, j]
    valid_indices = tl.where(mask, offsets, 0)
    
    # Calculate j and i from triangular indexing
    # For triangular matrix: index = (j-1)*j/2 + (i-1) for j >= 1, i >= 1, i <= j
    j_vals = tl.zeros_like(valid_indices)
    i_vals = tl.zeros_like(valid_indices)
    
    # Iterate through valid triangular positions
    for idx in range(BLOCK_SIZE):
        if idx + block_start < ((LEN_2D - 1) * LEN_2D) // 2:
            linear_idx = idx + block_start
            
            # Find j such that (j-1)*j/2 <= linear_idx < j*(j+1)/2
            j = 1
            cumsum = 0
            while cumsum + j <= linear_idx and j < LEN_2D:
                cumsum += j
                j += 1
            
            i = linear_idx - cumsum + 1
            
            # Read from copy array
            aa_copy_offset = j * LEN_2D + (i - 1)
            aa_prev_val = tl.load(aa_copy_ptr + aa_copy_offset, mask=(i > 1), other=0.0)
            
            bb_offset = j * LEN_2D + i
            bb_val = tl.load(bb_ptr + bb_offset)
            
            # Compute result
            result = aa_prev_val * aa_prev_val + bb_val
            
            # Store to original array
            aa_offset = j * LEN_2D + i
            tl.store(aa_ptr + aa_offset, result)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Calculate total number of triangular elements
    total_elements = (LEN_2D - 1) * LEN_2D // 2
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    s232_kernel[grid_size,](
        aa,
        aa_copy,
        bb,
        LEN_2D,
        BLOCK_SIZE,
    )