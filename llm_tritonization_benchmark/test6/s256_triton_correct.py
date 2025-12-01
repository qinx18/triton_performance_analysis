import torch
import triton
import triton.language as tl

@triton.jit
def s256_update_a_kernel(
    a_ptr, a_0, LEN_2D,
    BLOCK_SIZE: tl.constexpr
):
    """
    Kernel to update array a based on alternating pattern.
    a[even j] = a[0], a[odd j] = 1.0 - a[0]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < LEN_2D)

    # Compute a[j] based on whether j is odd or even
    is_odd = (offsets % 2) == 1
    a_vals = tl.where(is_odd, 1.0 - a_0, a_0)

    # Store a[j]
    tl.store(a_ptr + offsets, a_vals, mask=mask)


@triton.jit
def s256_compute_aa_kernel(
    aa_ptr, a_ptr, bb_ptr, d_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute aa[j, i] = a[j] + bb[j, i] * d[j]
    Each program processes one row j across multiple columns i.
    """
    row_j = tl.program_id(0)

    # Only process rows j >= 1
    if row_j < 1:
        return

    # Load a[j] and d[j] once for the entire row
    a_j = tl.load(a_ptr + row_j)
    d_j = tl.load(d_ptr + row_j)

    # Process columns in blocks
    for block_start in range(0, LEN_2D, BLOCK_SIZE):
        offsets_i = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets_i < LEN_2D

        # Compute indices for aa[row_j, offsets_i] and bb[row_j, offsets_i]
        indices = row_j * LEN_2D + offsets_i

        # Load bb[row_j, i]
        bb_vals = tl.load(bb_ptr + indices, mask=mask, other=0.0)

        # Compute aa[row_j, i] = a[row_j] + bb[row_j, i] * d[row_j]
        aa_vals = a_j + bb_vals * d_j

        # Store results
        tl.store(aa_ptr + indices, aa_vals, mask=mask)


def s256_triton(a, aa, bb, d):
    """
    Triton implementation of TSVC s256 - fully vectorized!

    Original pattern has apparent sequential dependency:
        for i in range(LEN_2D):
            for j in range(1, LEN_2D):
                a[j] = 1.0 - a[j - 1]
                aa[j][i] = a[j] + bb[j][i]*d[j]

    Key insight: a[j] alternates between two values:
        a[even j] = a[0]
        a[odd j] = 1.0 - a[0]

    This eliminates the dependency and allows full parallelization!

    Implementation:
    1. Vectorized update of a[] based on odd/even j
    2. Vectorized computation of aa[][] using updated a[]
    """
    a = a.contiguous()
    aa = aa.contiguous()
    bb = bb.contiguous()
    d = d.contiguous()

    LEN_2D = a.shape[0]

    # Get a[0] value
    a_0 = a[0].item()

    # Step 1: Update array a
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s256_update_a_kernel[grid](
        a, a_0, LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Step 2: Compute aa (one program per row)
    grid = (LEN_2D,)
    s256_compute_aa_kernel[grid](
        aa, a, bb, d,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a, aa
