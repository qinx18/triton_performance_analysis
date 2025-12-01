import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr, output_ptr,
    inc, j, LEN_2D, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s4116 - computes sum of a[inc+i] * aa[j-1][ip[i]]
    Each block processes BLOCK_SIZE elements and accumulates partial sums.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid indices
    mask = offsets < n_elements

    # Load ip values with masking
    ip_vals = tl.load(ip_ptr + offsets, mask=mask, other=0)

    # Calculate a array offsets: a[inc + i]
    a_offsets = inc + offsets
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)

    # Calculate aa array offsets: aa[j-1][ip[i]]
    # aa is 2D, offset = (j-1) * LEN_2D + ip[i]
    aa_row_start = (j - 1) * LEN_2D
    aa_offsets = aa_row_start + ip_vals
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)

    # Compute element-wise products and sum within block
    products = a_vals * aa_vals
    block_sum = tl.sum(products)

    # Store partial sum for this block
    tl.store(output_ptr + pid, block_sum)

def s4116_triton(a, aa, ip, inc, j):
    """
    Triton implementation of TSVC s4116 - indirect addressing with reduction.
    Uses block-based processing with reduction for GPU optimization.
    """
    a = a.contiguous()
    aa = aa.contiguous()
    ip = ip.contiguous()

    LEN_2D = aa.size(1)
    n_elements = LEN_2D - 1

    if n_elements <= 0:
        return a

    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)

    # Allocate output tensor for partial sums
    partial_sums = torch.zeros((grid_size,), device=a.device, dtype=a.dtype)

    # Launch kernel to compute partial sums
    s4116_kernel[(grid_size,)](
        a, aa, ip, partial_sums,
        inc, j, LEN_2D, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Final reduction of partial sums (done on GPU)
    sum_val = torch.sum(partial_sums)

    # Return original array as per baseline behavior
    return a
