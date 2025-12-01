import torch
import triton
import triton.language as tl

@triton.jit
def s252_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s252 - fully vectorizable!

    Pattern recognition:
        t = 0.0
        for i in range(N):
            s = b[i] * c[i]
            a[i] = s + t
            t = s

    Expands to:
        a[0] = b[0]*c[0] + 0.0 = b[0]*c[0]
        a[i] = b[i]*c[i] + b[i-1]*c[i-1] for i > 0

    This can be fully vectorized!
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load b and c values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)

    # Compute b[i] * c[i]
    bc_vals = b_vals * c_vals

    # For i=0, a[0] = b[0]*c[0]
    # For i>0, a[i] = b[i]*c[i] + b[i-1]*c[i-1]

    # Load b[i-1] * c[i-1] (shifted by 1)
    offsets_minus_1 = offsets - 1
    mask_minus_1 = (offsets > 0) & (offsets < n_elements)

    b_prev = tl.load(b_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)
    c_prev = tl.load(c_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)
    bc_prev = b_prev * c_prev

    # Compute a[i] = b[i]*c[i] + b[i-1]*c[i-1]
    # For i=0, bc_prev will be 0.0 due to mask
    a_vals = bc_vals + bc_prev

    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)


def s252_triton(a, b, c):
    """
    Triton implementation of TSVC s252 - fully vectorized!

    Original pattern has apparent sequential dependency:
        t = 0.0
        for i in range(N):
            s = b[i] * c[i]
            a[i] = s + t
            t = s

    But analysis shows:
        a[0] = b[0]*c[0]
        a[i] = b[i]*c[i] + b[i-1]*c[i-1] for i > 0

    This is fully vectorizable - each a[i] can be computed independently!
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    n_elements = a.numel()

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    s252_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a
