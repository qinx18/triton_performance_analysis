import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for unrolled dot product computation.
    Each thread block processes BLOCK_SIZE elements with manual unrolling.
    Writes partial sum to output buffer (cannot return values in Triton kernels).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask to handle array boundaries
    mask = offsets < n_elements

    # Load data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Compute element-wise products
    products = a_vals * b_vals

    # Reduce within block using tree reduction for better performance
    dot_partial = tl.sum(products)

    # Write partial result to output buffer
    tl.store(output_ptr + pid, dot_partial)

def s352_triton(a, b):
    """
    Triton implementation of TSVC s352 - unrolled dot product.
    Optimizes the dot product computation using GPU parallelization.

    Processes elements in groups of 5, matching the original C code behavior.
    Only processes complete groups of 5 - any remaining 1-4 elements are ignored.
    """
    a = a.contiguous()
    b = b.contiguous()

    # Only process complete groups of 5
    n_elements = a.numel()
    n_elements_to_process = (n_elements // 5) * 5

    if n_elements_to_process == 0:
        return a, b

    # Use block size that's efficient for GPU memory access
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements_to_process, BLOCK_SIZE),)

    # Create output tensor for partial results
    num_blocks = grid[0]
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)

    # Launch kernel to compute partial dot products
    s352_kernel[grid](
        a, b, partial_results, n_elements_to_process,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Sum up partial results to get final dot product
    # (Though the original function returns (a, b), not the dot product)
    dot = torch.sum(partial_results)

    return a, b
