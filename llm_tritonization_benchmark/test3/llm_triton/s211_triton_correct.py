import torch
import triton
import triton.language as tl

@triton.jit
def s211_kernel_middle(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, b_orig_ptr,
    start_idx,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for middle iterations (can be parallelized).

    For each i from start_idx to start_idx + n_elements - 1:
        b[i] = b_orig[i+1] - e[i] * d[i]
        a[i+1] = b[i] + c[i+1] * d[i+1]
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute actual indices
    i = start_idx + offsets
    mask = offsets < n_elements

    # Load b_orig[i+1], e[i], d[i] for computing b[i]
    b_next = tl.load(b_orig_ptr + i + 1, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i, mask=mask, other=0.0)

    # Compute b[i] = b_orig[i+1] - e[i] * d[i]
    b_new = b_next - e_vals * d_vals
    tl.store(b_ptr + i, b_new, mask=mask)

    # Load c[i+1], d[i+1] for computing a[i+1]
    c_next = tl.load(c_ptr + i + 1, mask=mask, other=0.0)
    d_next = tl.load(d_ptr + i + 1, mask=mask, other=0.0)

    # Compute a[i+1] = b[i] + c[i+1] * d[i+1]
    # Uses the just-computed b[i]
    a_new = b_new + c_next * d_next
    tl.store(a_ptr + i + 1, a_new, mask=mask)

def s211_triton(a, b, c, d, e):
    """
    Corrected Triton implementation of TSVC s211

    Uses loop regrouping to enable parallelization:
    - First:  a[1] = b[0] + c[1]*d[1]
    - Middle: b[i] = b[i+1] - e[i]*d[i]; a[i+1] = b[i] + c[i+1]*d[i+1]
              (parallelizable because b[i] and a[i+1] are independent indices)
    - Last:   b[n-2] = b[n-1] - e[n-2]*d[n-2]
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()

    n = a.shape[0]

    if n < 3:
        return a, b

    # Save original b values
    b_orig = b.clone()

    # First iteration: a[1] = b[0] + c[1]*d[1]
    a[1] = b[0] + c[1] * d[1]

    # Middle iterations: i from 1 to n-3 (parallelizable)
    if n > 3:
        n_middle = n - 3  # Number of middle iterations
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_middle, BLOCK_SIZE),)

        s211_kernel_middle[grid](
            a, b, c, d, e, b_orig,
            1,  # start_idx
            n_middle,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Last iteration: b[n-2] = b[n-1] - e[n-2]*d[n-2]
    b[n - 2] = b_orig[n - 1] - e[n - 2] * d[n - 2]

    return a, b
