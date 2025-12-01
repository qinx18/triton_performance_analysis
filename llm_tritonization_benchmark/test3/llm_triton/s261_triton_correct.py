import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel_main(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Main kernel for s261 - fully vectorizable!
    For i in [1, n_elements-2]:
        c[i] = c[i] * d[i]
        a[i+1] = a[i+1] + b[i+1] + c[i]_new

    No dependencies between iterations - fully parallel!
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets_i = block_start + tl.arange(0, BLOCK_SIZE) + 1  # i from 1 to n_elements-2
    mask = offsets_i < n_elements - 1

    # Load c[i] and d[i]
    c_vals = tl.load(c_ptr + offsets_i, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets_i, mask=mask, other=0.0)

    # Compute c[i] = c[i] * d[i]
    c_new = c_vals * d_vals

    # Store c[i]
    tl.store(c_ptr + offsets_i, c_new, mask=mask)

    # Load a[i+1] and b[i+1]
    a_vals = tl.load(a_ptr + offsets_i + 1, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets_i + 1, mask=mask, other=0.0)

    # Compute a[i+1] = a[i+1] + b[i+1] + c[i]_new
    a_new = a_vals + b_vals + c_new

    # Store a[i+1]
    tl.store(a_ptr + offsets_i + 1, a_new, mask=mask)


def s261_triton(a, b, c, d):
    """
    Triton implementation of TSVC s261 - fully vectorized!

    Original pattern has apparent sequential dependency:
        for i in range(1, LEN_1D):
            t = a[i] + b[i]
            a[i] = t + c[i-1]
            t = c[i] * d[i]
            c[i] = t

    Key insight: a[i] uses c[i-1] BEFORE c[i-1] is updated!
    Expansion:
        a[1] = a[1] + b[1] + c[0] (original c[0])
        a[2] = a[2] + b[2] + c[1]*d[1] (new c[1])
        a[i] = a[i] + b[i] + c[i-1]*d[i-1] for i >= 2

    Decomposition (all parallel):
    1. a[1] = a[1] + b[1] + c[0]
    2. For i in [1, LEN_1D-2]:
           c[i] = c[i] * d[i]
           a[i+1] = a[i+1] + b[i+1] + c[i]_new
    3. c[LEN_1D-1] = c[LEN_1D-1] * d[LEN_1D-1]
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()

    n_elements = a.shape[0]

    # Step 1: a[1] = a[1] + b[1] + c[0] (single element)
    a[1] = a[1] + b[1] + c[0]

    # Step 2: Main parallel loop for i in [1, n_elements-2]
    if n_elements > 2:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements - 2, BLOCK_SIZE),)
        s261_kernel_main[grid](
            a, b, c, d,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Step 3: c[LEN_1D-1] = c[LEN_1D-1] * d[LEN_1D-1] (single element)
    c[n_elements - 1] = c[n_elements - 1] * d[n_elements - 1]

    return a, c
