import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(
    bb_ptr, cc_ptr, flat_2d_array_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr
):
    """
    Corrected s126 kernel using same pattern as s1119.
    Each program handles BLOCK_SIZE columns in parallel.
    """
    # Get column block index
    col_block_id = tl.program_id(0)

    # Calculate column indices for this block
    col_start = col_block_id * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < LEN_2D

    # Process rows sequentially to maintain dependency
    for j in range(1, LEN_2D):
        # Calculate k indices for each column: k = 1 + i * LEN_2D + (j - 1)
        # where i is the column index
        k_indices = 1 + col_offsets * LEN_2D + (j - 1)

        # Load values
        bb_prev_offsets = (j - 1) * LEN_2D + col_offsets
        cc_curr_offsets = j * LEN_2D + col_offsets
        bb_curr_offsets = j * LEN_2D + col_offsets

        bb_prev = tl.load(bb_ptr + bb_prev_offsets, mask=col_mask, other=0.0)
        cc_val = tl.load(cc_ptr + cc_curr_offsets, mask=col_mask, other=0.0)
        flat_val = tl.load(flat_2d_array_ptr + k_indices - 1, mask=col_mask, other=0.0)

        # Compute new value: bb[j,i] = bb[j-1,i] + flat_2d_array[k-1] * cc[j,i]
        new_val = bb_prev + flat_val * cc_val

        # Store result
        tl.store(bb_ptr + bb_curr_offsets, new_val, mask=col_mask)


def s126_triton(bb, cc, flat_2d_array):
    """
    Corrected Triton implementation of s126.
    Uses same efficient grid pattern as s1119 for better performance.
    """
    bb = bb.contiguous()
    cc = cc.contiguous()
    flat_2d_array = flat_2d_array.contiguous()

    LEN_2D = bb.shape[0]

    # Use same strategy as s1119: launch fewer programs, each handling multiple columns
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)

    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return bb
