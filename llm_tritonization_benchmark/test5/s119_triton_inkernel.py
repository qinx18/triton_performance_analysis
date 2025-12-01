import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(
    aa_ptr,
    bb_ptr,
    M, N,
    stride_aa_0, stride_aa_1,
    stride_bb_0, stride_bb_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    S119 using s1119 pattern: single kernel, sequential row loop inside.

    Pattern: aa[i, j] = aa[i-1, j-1] + bb[i, j]
    """
    # Get column block index
    col_block_id = tl.program_id(0)

    # Calculate column indices for this block (j values)
    col_start = col_block_id * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)

    # Process rows sequentially INSIDE kernel (like s1119)
    for i in range(1, M):
        # Mask for valid columns (j >= 1 for this algorithm)
        # For aa[i, j], we need aa[i-1, j-1], so j must be >= 1
        col_mask = (col_offsets >= 1) & (col_offsets < N)

        # Calculate addresses for aa[i-1, j-1]
        aa_prev_addrs = aa_ptr + (i - 1) * stride_aa_0 + (col_offsets - 1) * stride_aa_1

        # Calculate addresses for aa[i, j]
        aa_curr_addrs = aa_ptr + i * stride_aa_0 + col_offsets * stride_aa_1

        # Calculate addresses for bb[i, j]
        bb_curr_addrs = bb_ptr + i * stride_bb_0 + col_offsets * stride_bb_1

        # Load values
        aa_prev = tl.load(aa_prev_addrs, mask=col_mask, other=0.0)
        bb_curr = tl.load(bb_curr_addrs, mask=col_mask, other=0.0)

        # Compute result
        result = aa_prev + bb_curr

        # Store result
        tl.store(aa_curr_addrs, result, mask=col_mask)


def s119_triton(aa, bb):
    """
    Triton implementation using s1119 pattern:
    - Single kernel launch
    - Sequential i-loop inside kernel
    - Parallel j processing
    """
    aa = aa.contiguous()
    bb = bb.contiguous()

    M, N = aa.shape

    if M <= 1 or N <= 1:
        return aa

    BLOCK_SIZE = 256

    # Launch ONE kernel (or few for column chunks)
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    grid = (grid_size,)

    s119_kernel[grid](
        aa, bb,
        M, N,
        aa.stride(0), aa.stride(1),
        bb.stride(0), bb.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return aa
