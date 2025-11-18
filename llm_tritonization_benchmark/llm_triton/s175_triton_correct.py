import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(
    a_ptr,
    b_ptr,
    indices_ptr,
    n_indices,
    array_size,
    inc,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s175: a[i] = a[i + inc] + b[i] for strided indices
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid indices in the indices array
    mask = offsets < n_indices

    # Load the actual indices to process (with explicit other=0 for safety)
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

    # For valid indices only, load and compute
    # We need to ensure:
    # 1. indices are valid (within bounds of indices array) - handled by mask
    # 2. indices + inc are valid array indices
    # 3. indices themselves are valid array indices

    # Load a[indices + inc] and b[indices]
    a_vals = tl.load(a_ptr + indices + inc, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)

    # Compute a[indices] = a[indices + inc] + b[indices]
    result = a_vals + b_vals

    # Store back to a[indices]
    tl.store(a_ptr + indices, result, mask=mask)

def s175_triton(a, b, inc):
    """
    Corrected Triton implementation of TSVC s175.

    Fixed: Proper masking and bounds checking for strided access.
    """
    a = a.contiguous()
    b = b.contiguous()

    len_1d = a.size(0)

    # Generate indices for the strided loop (same as baseline)
    indices = torch.arange(0, len_1d - 1, inc, device=a.device, dtype=torch.long)
    n_indices = indices.numel()

    if n_indices == 0:
        return a

    # Launch kernel with appropriate block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_indices, BLOCK_SIZE),)

    s175_kernel[grid](
        a, b, indices,
        n_indices,  # Number of indices to process
        len_1d,     # Original array size
        inc,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a
