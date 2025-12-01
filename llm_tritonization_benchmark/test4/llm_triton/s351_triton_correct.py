import torch
import triton
import triton.language as tl

@triton.jit
def s351_kernel(
    a_ptr,
    b_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for unrolled SAXPY operation.
    Processes elements in groups of 5, matching the original C unrolled loop.
    Only processes complete groups of 5.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for valid elements (only complete groups of 5)
    mask = offsets < n_elements

    # Load data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Perform SAXPY operation: a = a + alpha * b
    result = a_vals + alpha * b_vals

    # Store result back to memory
    tl.store(a_ptr + offsets, result, mask=mask)

def s351_triton(a, b, alpha):
    """
    Triton implementation of TSVC s351 - unrolled SAXPY operation.

    Processes elements in groups of 5, matching the original C code behavior.
    Only processes complete groups of 5 - any remaining 1-4 elements are left untouched.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()

    # Only process complete groups of 5
    n_elements = a.numel()
    n_elements_to_process = (n_elements // 5) * 5

    if n_elements_to_process == 0:
        return a

    # Use block size that balances occupancy and memory bandwidth
    BLOCK_SIZE = 1024

    # Calculate grid size for the elements we'll actually process
    grid = (triton.cdiv(n_elements_to_process, BLOCK_SIZE),)

    # Launch kernel
    s351_kernel[grid](
        a, b, alpha, n_elements_to_process,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a
