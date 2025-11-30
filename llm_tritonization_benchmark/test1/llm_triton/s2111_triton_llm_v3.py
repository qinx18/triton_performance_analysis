import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(
    aa_ptr,
    aa_copy_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which (j, i) position this program handles
    total_elements = (LEN_2D - 1) * (LEN_2D - 1)
    
    # Boundary check
    if pid >= total_elements:
        return
    
    # Convert linear index to 2D coordinates (j, i)
    # We need j from 1 to LEN_2D-1, i from 1 to LEN_2D-1
    linear_idx = pid
    i = (linear_idx % (LEN_2D - 1)) + 1  # i ranges from 1 to LEN_2D-1
    j = (linear_idx // (LEN_2D - 1)) + 1  # j ranges from 1 to LEN_2D-1
    
    # Calculate offsets for current position aa[j][i]
    curr_offset = j * LEN_2D + i
    
    # Calculate offsets for dependencies: aa[j][i-1] and aa[j-1][i]
    left_offset = j * LEN_2D + (i - 1)      # aa[j][i-1]
    above_offset = (j - 1) * LEN_2D + i     # aa[j-1][i]
    
    # Load values from read-only copy
    left_val = tl.load(aa_copy_ptr + left_offset)
    above_val = tl.load(aa_copy_ptr + above_offset)
    
    # Compute result
    result = (left_val + above_val) / 1.9
    
    # Store result to original array
    tl.store(aa_ptr + curr_offset, result)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Calculate total number of elements to process
    total_elements = (LEN_2D - 1) * (LEN_2D - 1)
    
    # Launch kernel with one thread per element
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s2111_kernel[grid](
        aa,
        aa_copy,
        LEN_2D,
        BLOCK_SIZE,
    )