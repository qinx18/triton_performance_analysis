import torch
import triton
import triton.language as tl

@triton.jit
def s255_kernel(
    a_ptr, b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s255 - fully vectorizable!

    Pattern analysis:
        x = b[N-1]; y = b[N-2]
        for i in range(N):
            a[i] = (b[i] + x + y) * 0.333
            y = x
            x = b[i]

    Expands to:
        a[0] = (b[0] + b[N-1] + b[N-2]) * 0.333
        a[1] = (b[1] + b[0] + b[N-1]) * 0.333
        a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333 for i >= 2

    This is fully vectorizable with special handling for i=0 and i=1!
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load b[i], b[i-1], b[i-2] values
    b_i = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # For b[i-1]: need wrap-around for i=0
    offsets_minus_1 = tl.where(offsets == 0, n_elements - 1, offsets - 1)
    b_i_minus_1 = tl.load(b_ptr + offsets_minus_1, mask=mask, other=0.0)

    # For b[i-2]: need wrap-around for i=0 (gives b[N-2]) and i=1 (gives b[N-1])
    offsets_minus_2 = tl.where(
        offsets == 0,
        n_elements - 2,
        tl.where(offsets == 1, n_elements - 1, offsets - 2)
    )
    b_i_minus_2 = tl.load(b_ptr + offsets_minus_2, mask=mask, other=0.0)

    # Compute a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333
    a_vals = (b_i + b_i_minus_1 + b_i_minus_2) * 0.333

    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)


def s255_triton(a, b):
    """
    Triton implementation of TSVC s255 - fully vectorized!

    Original pattern has apparent sequential dependency:
        x = b[N-1]; y = b[N-2]
        for i in range(N):
            a[i] = (b[i] + x + y) * 0.333
            y = x
            x = b[i]

    But analysis shows it's actually:
        a[0] = (b[0] + b[N-1] + b[N-2]) * 0.333
        a[1] = (b[1] + b[0] + b[N-1]) * 0.333
        a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333 for i >= 2

    This is fully vectorizable with proper edge case handling!
    """
    a = a.contiguous()
    b = b.contiguous()

    n_elements = b.size(0)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    s255_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a
