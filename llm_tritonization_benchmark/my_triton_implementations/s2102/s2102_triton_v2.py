import torch
import triton
import triton.language as tl

@triton.jit
def zero_kernel(
    aa_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to zero out all matrix elements.
    """
    pid = tl.program_id(0)

    # Calculate starting position for this block
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Zero out all elements in this block
    tl.store(aa_ptr + offsets, 0.0, mask=mask)

@triton.jit
def diagonal_kernel(
    aa_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to set diagonal elements to 1.
    Each thread processes one diagonal element.
    """
    pid = tl.program_id(0)

    # Calculate starting position for this block
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D

    # Calculate diagonal indices: diag[i] = i * LEN_2D + i
    diag_indices = offsets * LEN_2D + offsets

    # Set diagonal elements to 1.0
    tl.store(aa_ptr + diag_indices, 1.0, mask=mask)

def s2102_triton(aa):
    """
    Triton implementation of TSVC s2102 function.

    Two-kernel approach:
    1. First kernel zeros out all elements in parallel
    2. Second kernel sets diagonal elements to 1 in parallel

    This avoids sequential processing of diagonal elements.
    """
    aa = aa.contiguous()
    LEN_2D = aa.size(0)

    # Calculate total number of elements
    total_elements = LEN_2D * LEN_2D

    # Choose block size for optimal occupancy
    BLOCK_SIZE = 256

    # Launch first kernel to zero out all elements
    num_blocks_zero = triton.cdiv(total_elements, BLOCK_SIZE)
    zero_kernel[(num_blocks_zero,)](
        aa,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Launch second kernel to set diagonal elements to 1
    num_blocks_diag = triton.cdiv(LEN_2D, BLOCK_SIZE)
    diagonal_kernel[(num_blocks_diag,)](
        aa,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return aa
