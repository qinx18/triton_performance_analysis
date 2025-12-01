import torch
import triton
import triton.language as tl

@triton.jit
def s257_compute_a_kernel(
    a_ptr, aa_ptr,
    LEN_2D,
):
    """
    Kernel to compute a[i] = aa[LEN_2D-1][i] - a[i-1]
    Must be sequential due to dependency on a[i-1].
    Only one program runs this kernel.
    """
    # Sequential computation of a[i] using last row of aa
    for i in range(1, LEN_2D):
        # Load a[i-1]
        a_prev = tl.load(a_ptr + i - 1)

        # Load aa[LEN_2D-1][i] (last row, column i)
        aa_last_row = tl.load(aa_ptr + (LEN_2D - 1) * LEN_2D + i)

        # Compute a[i] = aa[LEN_2D-1][i] - a[i-1]
        a_new = aa_last_row - a_prev

        # Store a[i]
        tl.store(a_ptr + i, a_new)


@triton.jit
def s257_compute_aa_kernel(
    aa_ptr, a_ptr, bb_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute aa[j][i] = aa[j][i] - a[i-1] + bb[j][i]
    Fully vectorizable - no dependencies!
    Each program processes one row j.
    """
    row_j = tl.program_id(0)

    # Process columns in blocks
    for block_start in range(1, LEN_2D, BLOCK_SIZE):
        offsets_i = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets_i < LEN_2D

        # Compute indices for aa[row_j, i] and bb[row_j, i]
        indices = row_j * LEN_2D + offsets_i

        # Load aa[row_j, i] (original values)
        aa_vals = tl.load(aa_ptr + indices, mask=mask, other=0.0)

        # Load bb[row_j, i]
        bb_vals = tl.load(bb_ptr + indices, mask=mask, other=0.0)

        # Load a[i-1]
        a_prev_vals = tl.load(a_ptr + offsets_i - 1, mask=mask, other=0.0)

        # Compute aa[row_j, i] = aa[row_j, i] - a[i-1] + bb[row_j, i]
        aa_new = aa_vals - a_prev_vals + bb_vals

        # Store results
        tl.store(aa_ptr + indices, aa_new, mask=mask)


def s257_triton(a, aa, bb):
    """
    Triton implementation of TSVC s257 - array expansion technique!

    Original pattern has apparent sequential dependency:
        for i in range(1, LEN_2D):
            for j in range(LEN_2D):
                a[i] = aa[j][i] - a[i-1]
                aa[j][i] = a[i] + bb[j][i]

    Key insight: a[i] is overwritten LEN_2D times, only the last write matters!
        Final value: a[i] = aa[LEN_2D-1][i] - a[i-1]

    Analysis:
        a[i] after all j iterations = aa[LEN_2D-1][i] - a[i-1]
        aa[j][i] = (aa_orig[j][i] - a[i-1]) + bb[j][i]

    Implementation:
    1. Sequential kernel: Compute a[i] = aa[LEN_2D-1][i] - a[i-1]
    2. Parallel kernel: Compute aa[j][i] = aa[j][i] - a[i-1] + bb[j][i]
    """
    a = a.contiguous()
    aa = aa.contiguous()
    bb = bb.contiguous()

    LEN_2D = aa.shape[0]

    # Step 1: Compute a[i] sequentially (single program)
    grid = (1,)
    s257_compute_a_kernel[grid](
        a, aa,
        LEN_2D,
    )

    # Step 2: Compute aa[j][i] in parallel (one program per row)
    BLOCK_SIZE = 256
    grid = (LEN_2D,)
    s257_compute_aa_kernel[grid](
        aa, a, bb,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a, aa
