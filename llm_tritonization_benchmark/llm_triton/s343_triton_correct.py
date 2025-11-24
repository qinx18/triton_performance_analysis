import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(
    aa_ptr, bb_ptr, flat_2d_array_ptr, prefix_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for conditional array packing (2D to 1D).
    Uses prefix sum to determine output positions for positive values.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values from bb and aa
    bb_vals = tl.load(bb_ptr + offsets, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask, other=0.0)

    # Check condition bb > 0
    condition = bb_vals > 0.0

    # Load prefix sum values to determine output positions
    prefix_vals = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0)

    # For elements that satisfy condition, write to output array
    # Prefix sum gives us the 1-based position, subtract 1 for 0-based indexing
    output_mask = mask & condition
    output_offsets = prefix_vals - 1

    # Only write if condition is met and we have a valid position
    valid_write_mask = output_mask & (output_offsets >= 0)

    tl.store(flat_2d_array_ptr + output_offsets, aa_vals, mask=valid_write_mask)

def s343_triton(aa, bb, flat_2d_array):
    """
    Triton implementation of TSVC s343 - conditional array packing.
    Optimized using prefix sum to determine output positions in parallel.

    Args:
        aa: 2D tensor (read-only)
        bb: 2D tensor (read-only)
        flat_2d_array: 1D tensor (read-write)

    Returns:
        flat_2d_array: Modified 1D tensor
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    flat_2d_array = flat_2d_array.contiguous()

    # Flatten 2D arrays to 1D for processing
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    n_elements = aa_flat.numel()

    if n_elements == 0:
        return flat_2d_array

    # Create mask for condition bb > 0
    condition_mask = (bb_flat > 0.0).int()

    # Compute prefix sum to determine output positions
    prefix_sum = torch.cumsum(condition_mask, dim=0)

    # Launch kernel with appropriate block size
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    s343_kernel[grid](
        aa_flat, bb_flat, flat_2d_array, prefix_sum,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return flat_2d_array
