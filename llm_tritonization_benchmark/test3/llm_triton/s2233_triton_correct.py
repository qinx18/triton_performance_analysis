import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel_aa(aa_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for aa part of s2233: aa[j][i] = aa[j-1][i] + cc[j][i]
    Each program processes one column (same as s233/s231).
    Vertical dependency: must process rows sequentially within each column.
    """
    # Each program handles one column (skip column 0, start from column 1)
    col_idx = tl.program_id(0) + 1

    # Process rows sequentially within this column
    for j in range(1, LEN_2D):
        # Calculate memory addresses
        curr_addr = aa_ptr + j * LEN_2D + col_idx
        prev_addr = aa_ptr + (j - 1) * LEN_2D + col_idx
        cc_addr = cc_ptr + j * LEN_2D + col_idx

        # Load values
        aa_prev = tl.load(prev_addr)
        cc_curr = tl.load(cc_addr)

        # Compute and store result
        result = aa_prev + cc_curr
        tl.store(curr_addr, result)


@triton.jit
def s2233_kernel_bb(bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for bb part of s2233: bb[i][j] = bb[i-1][j] + cc[i][j]
    Each program processes one column (different from s233!).
    Vertical dependency: must process rows sequentially within each column.

    Key difference from s233:
    - s233: bb[j][i] = bb[j][i-1] + cc[j][i] -> horizontal, process by row
    - s2233: bb[i][j] = bb[i-1][j] + cc[i][j] -> vertical, process by column
    """
    # Each program handles one column (skip column 0, start from column 1)
    col_idx = tl.program_id(0) + 1

    # Process rows sequentially within this column
    for i in range(1, LEN_2D):
        # Calculate memory addresses
        curr_addr = bb_ptr + i * LEN_2D + col_idx
        prev_addr = bb_ptr + (i - 1) * LEN_2D + col_idx
        cc_addr = cc_ptr + i * LEN_2D + col_idx

        # Load values
        bb_prev = tl.load(prev_addr)
        cc_curr = tl.load(cc_addr)

        # Compute and store result
        result = bb_prev + cc_curr
        tl.store(curr_addr, result)


def s2233_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s2233 with split kernels.

    Original pattern:
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[j][i] = aa[j-1][i] + cc[j][i];  // vertical dependency
            }
            for (int j = 1; j < LEN_2D; j++) {
                bb[i][j] = bb[i-1][j] + cc[i][j];  // vertical dependency (different from s233!)
            }
        }

    Split into two kernels:
    - aa_kernel: ONE launch with LEN_2D-1 programs, each processing one column
    - bb_kernel: ONE launch with LEN_2D-1 programs, each processing one column

    Key insight for s2233:
    - Both aa and bb have vertical dependencies (process by column)
    - aa[j][i] depends on aa[j-1][i], so columns are independent but rows are sequential
    - bb[i][j] depends on bb[i-1][j], so columns are independent but rows are sequential
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()

    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128

    # Process aa: one program per column (skip column 0, process columns 1 to LEN_2D-1)
    grid = (LEN_2D - 1,)
    s2233_kernel_aa[grid](
        aa, cc, LEN_2D, BLOCK_SIZE
    )

    # Process bb: one program per column (skip column 0, process columns 1 to LEN_2D-1)
    # This is different from s233 where bb was processed by row!
    s2233_kernel_bb[grid](
        bb, cc, LEN_2D, BLOCK_SIZE
    )

    return aa, bb
