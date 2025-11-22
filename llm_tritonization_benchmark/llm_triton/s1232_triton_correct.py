import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel_row(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s1232: aa[i][j] = bb[i][j] + cc[i][j] where i >= j
    Each program processes one complete row i, handling columns j = 0 to i with in-kernel loop.
    """
    # Each program handles one row i
    row_idx = tl.program_id(0)

    # Process columns in chunks of BLOCK_SIZE
    # For row i, we need to process columns 0 to i (inclusive)
    num_cols = row_idx + 1  # Number of columns to process for this row

    for col_start in range(0, num_cols, BLOCK_SIZE):
        # Calculate column offsets for this chunk
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)

        # Mask for valid columns (must be <= row_idx and < LEN_2D)
        mask = (col_offsets <= row_idx) & (col_offsets < LEN_2D)

        # Calculate linear indices for this row: row_idx * LEN_2D + col
        linear_indices = row_idx * LEN_2D + col_offsets

        # Load, compute, and store (vectorized)
        bb_vals = tl.load(bb_ptr + linear_indices, mask=mask, other=0.0)
        cc_vals = tl.load(cc_ptr + linear_indices, mask=mask, other=0.0)
        result = bb_vals + cc_vals
        tl.store(aa_ptr + linear_indices, result, mask=mask)


def s1232_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s1232 with one-program-per-row.

    Single kernel launch where each program processes one complete row.
    No data dependencies, so everything runs in parallel.

    Original pattern (can be fully parallelized):
        for (int j = 0; j < LEN_2D; j++) {
            for (int i = j; i < LEN_2D; i++) {
                aa[i][j] = bb[i][j] + cc[i][j];
            }
        }

    Equivalent to (one program per row i):
        for (int i = 0; i < LEN_2D; i++) {        # parallel - each program handles one i
            for (int j = 0; j <= i; j++) {        # vectorized - handled in parallel by program
                aa[i][j] = bb[i][j] + cc[i][j];
            }
        }

    Single kernel launch with LEN_2D programs.
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()

    LEN_2D = aa.shape[0]

    # Launch one program per row
    BLOCK_SIZE = 1024  # Must be >= LEN_2D to handle all columns in one vector load
    grid = (LEN_2D,)

    s1232_kernel_row[grid](
        aa, bb, cc, LEN_2D, BLOCK_SIZE
    )

    return aa
