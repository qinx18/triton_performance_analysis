import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(
    aa_ptr,
    bb_ptr,
    cc_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for diagonal matrix multiplication.
    Each thread block processes BLOCK_SIZE diagonal elements.
    """
    # Get the current block's starting position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases where offsets exceed matrix size
    mask = offsets < n
    
    # Calculate linear indices for diagonal elements: i * n + i = i * (n + 1)
    diagonal_indices = offsets * (n + 1)
    
    # Load diagonal elements from bb and cc matrices
    bb_diag = tl.load(bb_ptr + diagonal_indices, mask=mask)
    cc_diag = tl.load(cc_ptr + diagonal_indices, mask=mask)
    
    # Load current diagonal elements from aa matrix
    aa_diag = tl.load(aa_ptr + diagonal_indices, mask=mask)
    
    # Compute: aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_diag + bb_diag * cc_diag
    
    # Store the updated diagonal elements back to aa matrix
    tl.store(aa_ptr + diagonal_indices, result, mask=mask)

def s2101_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s2101 - diagonal matrix multiplication.
    Optimized for GPU execution with efficient memory coalescing.
    """
    # Ensure tensors are contiguous for optimal memory access
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    n = aa.size(0)
    
    # Choose block size for good occupancy - power of 2 for efficiency
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed to cover all diagonal elements
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    s2101_kernel[grid](
        aa,
        bb, 
        cc,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return aa