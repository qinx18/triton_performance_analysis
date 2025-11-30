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
    
    # Calculate total number of elements in upper triangle (excluding diagonal for j=0)
    # For j from 1 to LEN_2D-1, i from 1 to j
    total_elements = (LEN_2D - 1) * LEN_2D // 2
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert linear index to (j, i) coordinates
    # We need to find j and i such that the linear index maps to the upper triangle
    valid_offsets = tl.where(mask, offsets, 0)
    
    # For upper triangle starting at j=1, i=1
    # Linear index = sum from k=1 to j-1 of k + (i-1) = j*(j-1)/2 + (i-1)
    # Given linear index, find j: j = ceil((-1 + sqrt(1 + 8*idx)) / 2) + 1
    
    # Approximate j using the quadratic formula
    temp = 1.0 + 8.0 * valid_offsets.to(tl.float32)
    sqrt_temp = tl.sqrt(temp)
    j_approx = (-1.0 + sqrt_temp) / 2.0
    j = j_approx.to(tl.int32) + 1
    
    # Ensure j is at least 1 and at most LEN_2D-1
    j = tl.maximum(j, 1)
    j = tl.minimum(j, LEN_2D - 1)
    
    # Calculate i based on j and linear index
    # linear_idx = j*(j-1)/2 + (i-1)
    # i = linear_idx - j*(j-1)/2 + 1
    triangle_start = j * (j - 1) // 2
    i = valid_offsets - triangle_start + 1
    
    # Ensure i is in valid range [1, j]
    i = tl.maximum(i, 1)
    i = tl.minimum(i, j)
    
    # Calculate 2D array indices
    aa_idx = j * LEN_2D + i
    aa_prev_idx = j * LEN_2D + (i - 1)
    bb_idx = j * LEN_2D + i
    
    # Bounds check for array access
    valid_access = (j >= 1) & (j < LEN_2D) & (i >= 1) & (i <= j) & mask
    
    # Load values
    aa_prev = tl.load(aa_copy_ptr + aa_prev_idx, mask=valid_access, other=0.0)
    bb_val = tl.load(bb_ptr + bb_idx, mask=valid_access, other=0.0)
    
    # Compute result
    result = aa_prev * aa_prev + bb_val
    
    # Store result
    tl.store(aa_ptr + aa_idx, result, mask=valid_access)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Calculate total number of elements in upper triangle
    total_elements = (LEN_2D - 1) * LEN_2D // 2
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    s232_kernel[(grid_size,)](
        aa,
        aa_copy,
        bb,
        LEN_2D,
        BLOCK_SIZE,
    )