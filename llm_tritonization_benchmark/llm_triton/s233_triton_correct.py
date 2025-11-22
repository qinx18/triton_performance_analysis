import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel_aa(aa_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for aa part of s233: aa[j][i] = aa[j-1][i] + cc[j][i]
    Each program processes one column (like s231).
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
def s233_kernel_bb(bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for bb part of s233: bb[j][i] = bb[j][i-1] + cc[j][i]
    Each program processes one row (horizontal pattern).
    Horizontal dependency: must process columns sequentially within each row.
    """
    # Each program handles one row (skip row 0, start from row 1)
    row_idx = tl.program_id(0) + 1

    # Process columns sequentially within this row
    for i in range(1, LEN_2D):
        # Calculate memory addresses
        curr_addr = bb_ptr + row_idx * LEN_2D + i
        prev_addr = bb_ptr + row_idx * LEN_2D + (i - 1)
        cc_addr = cc_ptr + row_idx * LEN_2D + i

        # Load values
        bb_prev = tl.load(prev_addr)
        cc_curr = tl.load(cc_addr)

        # Compute and store result
        result = bb_prev + cc_curr
        tl.store(curr_addr, result)


def s233_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s233 with split kernels.

    Original pattern:
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[j][i] = aa[j-1][i] + cc[j][i];  // vertical dependency
            }
            for (int j = 1; j < LEN_2D; j++) {
                bb[j][i] = bb[j][i-1] + cc[j][i];  // horizontal dependency
            }
        }

    Split into two kernels (2 x s231 pattern):
    - aa_kernel: ONE launch with LEN_2D programs, each processing one column
    - bb_kernel: ONE launch with LEN_2D programs, each processing one row
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()

    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128

    # Process aa: one program per column (skip column 0, process columns 1 to LEN_2D-1)
    grid = (LEN_2D - 1,)
    s233_kernel_aa[grid](
        aa, cc, LEN_2D, BLOCK_SIZE
    )

    # Process bb: one program per row (skip row 0, process rows 1 to LEN_2D-1)
    s233_kernel_bb[grid](
        bb, cc, LEN_2D, BLOCK_SIZE
    )

    return aa, bb
