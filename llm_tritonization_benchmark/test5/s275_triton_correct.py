import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D,
):
    """
    Kernel for s275 - one program per column.
    Each program handles one column sequentially if aa[0][i] > 0.
    """
    # Get column index
    col_i = tl.program_id(0)

    # Check if aa[0][i] > 0
    aa_0_i = tl.load(aa_ptr + 0 * LEN_2D + col_i)

    if aa_0_i > 0.0:
        # Sequentially update column i
        for j in range(1, LEN_2D):
            # Load aa[j-1][i]
            aa_prev = tl.load(aa_ptr + (j - 1) * LEN_2D + col_i)

            # Load bb[j][i] and cc[j][i]
            bb_val = tl.load(bb_ptr + j * LEN_2D + col_i)
            cc_val = tl.load(cc_ptr + j * LEN_2D + col_i)

            # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
            aa_new = aa_prev + bb_val * cc_val

            # Store aa[j][i]
            tl.store(aa_ptr + j * LEN_2D + col_i, aa_new)


def s275_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s275 - conditional column updates!

    Original pattern:
        for i in range(LEN_2D):
            if (aa[0][i] > 0.):
                for j in range(1, LEN_2D):
                    aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]

    Key insight: Each column is independent!
        - For each column i where aa[0][i] > 0, update column sequentially
        - Different columns can be processed in parallel
        - Within each column, updates are sequential (dependency on aa[j-1][i])

    Implementation:
        - Launch one program per column (LEN_2D programs)
        - Each program checks condition and processes its column sequentially
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()

    LEN_2D = aa.shape[1]

    # Launch one program per column
    grid = (LEN_2D,)
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D,
    )

    return aa
