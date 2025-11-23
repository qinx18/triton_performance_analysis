import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(
    aa_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to zero out matrix and set diagonal to 1.
    Uses 1D indexing for efficient memory access.
    """
    pid = tl.program_id(0)

    # Calculate starting position for this block
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D * LEN_2D

    # Zero out all elements in this block
    tl.store(aa_ptr + offsets, 0.0, mask=mask)

    # Check if any diagonal elements are in this block
    # Diagonal elements are at positions: 0, LEN_2D+1, 2*(LEN_2D+1), ...
    for i in range(LEN_2D):
        diag_idx = i * LEN_2D + i  # Row i, column i
        if (start_idx <= diag_idx) and (diag_idx < start_idx + BLOCK_SIZE):
            # This diagonal element is in our block
            local_offset = diag_idx - start_idx
            if local_offset < BLOCK_SIZE:
                tl.store(aa_ptr + diag_idx, 1.0)

def s2102_triton(aa):
    """
    Triton implementation of TSVC s2102 function.

    Optimizations:
    - Single kernel launch processes entire matrix
    - 1D memory access pattern for better coalescing
    - Block-wise processing with masking for edge cases
    - In-place diagonal setting within the same kernel
    """
    aa = aa.contiguous()
    LEN_2D = aa.size(0)

    # Calculate total number of elements
    total_elements = LEN_2D * LEN_2D

    # Choose block size for optimal occupancy
    BLOCK_SIZE = 256

    # Calculate number of blocks needed
    num_blocks = triton.cdiv(total_elements, BLOCK_SIZE)

    # Launch kernel
    s2102_kernel[(num_blocks,)](
        aa,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return aa
