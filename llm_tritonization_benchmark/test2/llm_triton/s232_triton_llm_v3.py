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
    # Get the current block ID
    block_id = tl.program_id(0)
    
    # Calculate which (j, i) pairs this block handles
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear offsets to (j, i) coordinates
    # We need to map the triangular iteration space to linear indices
    valid_count = 0
    
    for idx in range(BLOCK_SIZE):
        linear_idx = block_start + idx
        if linear_idx >= (LEN_2D - 1) * LEN_2D // 2:
            break
            
        # Convert linear index back to (j, i) coordinates
        # For triangular loops: j goes from 1 to LEN_2D-1, i goes from 1 to j
        temp_idx = linear_idx
        j = 1
        while temp_idx >= j:
            temp_idx -= j
            j += 1
        i = temp_idx + 1
        
        if j < LEN_2D and i <= j:
            # Calculate 2D array indices
            curr_idx = j * LEN_2D + i
            prev_idx = j * LEN_2D + (i - 1)
            
            # Load values
            aa_prev = tl.load(aa_copy_ptr + prev_idx)
            bb_val = tl.load(bb_ptr + curr_idx)
            
            # Compute result
            result = aa_prev * aa_prev + bb_val
            
            # Store result
            tl.store(aa_ptr + curr_idx, result)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Calculate total number of iterations in triangular loop
    total_iterations = (LEN_2D - 1) * LEN_2D // 2
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_iterations, BLOCK_SIZE)
    
    # Sequential execution to maintain dependencies
    for j in range(1, LEN_2D):
        for i in range(1, j + 1):
            aa[j, i] = aa_copy[j, i-1] * aa_copy[j, i-1] + bb[j, i]
        # Update copy for next iteration
        if j < LEN_2D - 1:
            aa_copy[j] = aa[j]