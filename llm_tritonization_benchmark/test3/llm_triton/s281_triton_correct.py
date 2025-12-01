import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel_first_half(
    a_ptr, b_ptr, c_ptr, a_temp_ptr,
    n_elements, mid,
    BLOCK_SIZE: tl.constexpr,
):
    """
    First half: Process i in [0, mid-1]
    Read from original a, write to both a and a_temp
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < mid)

    # Compute reverse indices: LEN_1D - i - 1
    reverse_offsets = n_elements - 1 - offsets

    # Load a[LEN_1D-i-1] (from original a), b[i], c[i]
    a_rev = tl.load(a_ptr + reverse_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)

    # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
    x = a_rev + b_vals * c_vals

    # Store a[i] = x - 1.0
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)

    # Store to temp for second half to read
    tl.store(a_temp_ptr + offsets, x - 1.0, mask=mask)

    # Store b[i] = x
    tl.store(b_ptr + offsets, x, mask=mask)


@triton.jit
def s281_kernel_second_half(
    a_ptr, b_ptr, c_ptr, a_temp_ptr,
    n_elements, mid,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Second half: Process i in [mid, n_elements-1]
    Read from a_temp (updated values), write to a
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + mid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n_elements)

    # Compute reverse indices: LEN_1D - i - 1
    reverse_offsets = n_elements - 1 - offsets

    # Load a[LEN_1D-i-1] from TEMP (updated values), b[i], c[i]
    a_rev = tl.load(a_temp_ptr + reverse_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)

    # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
    x = a_rev + b_vals * c_vals

    # Store a[i] = x - 1.0
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)

    # Store b[i] = x
    tl.store(b_ptr + offsets, x, mask=mask)


def s281_triton(a, b, c):
    """
    Triton implementation of TSVC s281 - handles read-after-write dependency!

    Original pattern has dependency:
        for i in range(LEN_1D):
            x = a[LEN_1D-i-1] + b[i] * c[i]
            a[i] = x - 1.0
            b[i] = x

    Key insight: At i >= LEN_1D/2, we read a[LEN_1D-i-1] which was written earlier!
        - i=0: Read a[LEN_1D-1] (original), Write a[0]
        - ...
        - i=LEN_1D/2: Read a[LEN_1D/2-1] (UPDATED!), Write a[LEN_1D/2]

    Solution: Split into two phases
        1. First half [0, mid-1]: Read original a, write to temp array
        2. Second half [mid, LEN_1D-1]: Read from temp array (updated values)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    n_elements = a.shape[0]
    mid = n_elements // 2

    # Create temp array to store updated a values
    a_temp = a.clone()

    BLOCK_SIZE = 256

    # Phase 1: Process first half [0, mid-1]
    grid_first = (triton.cdiv(mid, BLOCK_SIZE),)
    s281_kernel_first_half[grid_first](
        a, b, c, a_temp,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Phase 2: Process second half [mid, n_elements-1]
    remaining = n_elements - mid
    grid_second = (triton.cdiv(remaining, BLOCK_SIZE),)
    s281_kernel_second_half[grid_second](
        a, b, c, a_temp,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a, b
