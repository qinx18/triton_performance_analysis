import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(
    sparse_ptr, cond_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    half_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    First pass: compute values and store sparsely with condition flags.
    Each input i produces 2 potential outputs at positions 2*i and 2*i+1.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < half_len

    # Load input values
    b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_val = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_val = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_val = tl.load(e_ptr + offsets, mask=mask, other=0.0)

    # Compute common expression
    de_product = d_val * e_val

    # First value: always written at position 2*i
    first_val = b_val + de_product
    output_idx = 2 * offsets
    tl.store(sparse_ptr + output_idx, first_val, mask=mask)

    # Second value: conditionally written at position 2*i+1
    # Use tl.where instead of if tl.any
    cond_mask = c_val > 0.0
    second_val = tl.where(cond_mask, c_val + de_product, 0.0)
    tl.store(sparse_ptr + output_idx + 1, second_val, mask=mask)

    # Store the condition flag (1.0 if c > 0, 0.0 otherwise)
    cond_flag = tl.where(cond_mask, 1.0, 0.0)
    tl.store(cond_ptr + offsets, cond_flag, mask=mask)


@triton.jit
def s123_compact_kernel(
    a_ptr, sparse_ptr, cond_ptr,
    half_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Second pass: compact the sparse output using the condition flags.
    This runs sequentially on one thread for correctness.
    """
    # Only thread 0 does the compaction
    if tl.program_id(0) != 0:
        return

    write_pos = 0

    # Process each input element
    for i in range(half_len):
        # Always copy first element (from position 2*i)
        val1 = tl.load(sparse_ptr + 2 * i)
        tl.store(a_ptr + write_pos, val1)
        write_pos += 1

        # Check condition flag to see if second element should be copied
        cond = tl.load(cond_ptr + i)
        if cond > 0.5:  # Flag is 1.0 if condition was true
            val2 = tl.load(sparse_ptr + 2 * i + 1)
            tl.store(a_ptr + write_pos, val2)
            write_pos += 1


def s123_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s123 - conditional assignment with variable indexing.

    Two-pass approach:
    1. First kernel: Compute values and store sparsely with condition flags
    2. Second kernel: Compact based on condition flags (not value == 0 check)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()

    len_1d = b.shape[0] * 2
    half_len = len_1d // 2

    # Create temporary arrays
    sparse_a = torch.zeros(len_1d, dtype=a.dtype, device=a.device)
    cond_flags = torch.zeros(half_len, dtype=a.dtype, device=a.device)

    # First pass: compute values with sparse storage
    BLOCK_SIZE = 256
    grid = (triton.cdiv(half_len, BLOCK_SIZE),)

    s123_kernel[grid](
        sparse_a, cond_flags, b, c, d, e,
        half_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Second pass: compact using condition flags
    # Launch single thread for sequential compaction
    s123_compact_kernel[(1,)](
        a, sparse_a, cond_flags,
        half_len,
        BLOCK_SIZE=1,
    )

    return a
