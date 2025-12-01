import torch
import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create offsets for the block (unrolled by 5)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to handle edge cases (only complete groups of 5)
    mask = offsets < n_elements

    # Load indices from ip array with masking
    ip_vals = tl.load(ip_ptr + offsets, mask=mask, other=0)

    # Ensure ip_vals are within bounds of b array
    ip_mask = (ip_vals >= 0) & (ip_vals < n_elements) & mask

    # Load values from b array using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=ip_mask, other=0.0)

    # Load current values from a array
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)

    # Perform saxpy operation: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals

    # Store results back to a array
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, ip, alpha):
    """
    Triton implementation of TSVC s353 - unrolled saxpy with indirect addressing.

    Processes elements in groups of 5, matching the original C code behavior.
    Only processes complete groups of 5 - any remaining 1-4 elements are left untouched.
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()

    LEN_1D = a.shape[0]

    # Only process complete groups of 5
    n_elements_to_process = (LEN_1D // 5) * 5

    if n_elements_to_process == 0:
        return a

    # Choose block size for good memory coalescing
    BLOCK_SIZE = 256

    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements_to_process, meta['BLOCK_SIZE']),)

    # Launch kernel
    s353_kernel[grid](
        a, b, ip, alpha, n_elements_to_process,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return a
