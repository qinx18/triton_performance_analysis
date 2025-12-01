import triton
import triton.language as tl
import torch

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate starting positions
    i_start = pid_i * BLOCK_SIZE
    j_start = pid_j * BLOCK_SIZE
    
    # Create offset arrays once at the start
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    # Create masks
    i_mask = i_indices < LEN_2D
    j_mask = j_indices < LEN_2D
    
    # Create 2D index arrays for memory access
    i_indices_2d = i_indices[:, None]
    j_indices_2d = j_indices[None, :]
    
    # Calculate memory addresses for the block
    indices = j_indices_2d * LEN_2D + i_indices_2d
    
    # Create 2D mask
    mask_2d = i_mask[:, None] & j_mask[None, :] & (indices < LEN_2D * LEN_2D)
    
    # Initialize block to zeros
    zeros = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Store zeros to the entire block
    tl.store(aa_ptr + indices, zeros, mask=mask_2d)
    
    # Now handle diagonal elements
    # For each valid i in this block, check if the diagonal element (i,i) is in range
    for i_offset in range(BLOCK_SIZE):
        i_idx = i_start + i_offset
        if i_idx < LEN_2D:
            # Check if diagonal element (i_idx, i_idx) falls within our j block range
            if j_start <= i_idx < j_start + BLOCK_SIZE:
                j_offset = i_idx - j_start
                # Calculate the memory address for aa[i_idx][i_idx]
                diag_addr = i_idx * LEN_2D + i_idx
                tl.store(aa_ptr + diag_addr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 16
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(LEN_2D, BLOCK_SIZE)
    grid_j = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Launch kernel
    s2102_kernel[(grid_i, grid_j)](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )