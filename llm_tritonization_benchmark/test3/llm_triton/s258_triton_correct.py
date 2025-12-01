import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(
    s_array_ptr, a_ptr, d_ptr,
    n_elements,
):
    """
    Kernel to expand scalar s into array.
    s[i] = d[i] * d[i] if a[i] > 0, else s[i-1]
    Must be sequential due to dependency on s[i-1].
    """
    # Initialize s = 0
    s = 0.0

    # Sequential computation
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)

        # Update s if a[i] > 0
        if a_val > 0.0:
            d_val = tl.load(d_ptr + i)
            s = d_val * d_val

        # Store s[i]
        tl.store(s_array_ptr + i, s)


@triton.jit
def s258_compute_kernel(
    b_ptr, e_ptr, s_array_ptr, c_ptr, d_ptr, aa_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute b[i] and e[i] using expanded s array.
    Fully vectorizable!
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    s_vals = tl.load(s_array_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask, other=0.0)

    # Compute b[i] = s[i] * c[i] + d[i]
    b_vals = s_vals * c_vals + d_vals

    # Compute e[i] = (s[i] + 1.0) * aa[0][i]
    e_vals = (s_vals + 1.0) * aa_vals

    # Store results
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    tl.store(e_ptr + offsets, e_vals, mask=mask)


def s258_triton(a, aa, b, c, d, e):
    """
    Triton implementation of TSVC s258 - scalar expansion!

    Original pattern has sequential dependency:
        s = 0.
        for i in range(LEN_2D):
            if (a[i] > 0.):
                s = d[i] * d[i]
            b[i] = s * c[i] + d[i]
            e[i] = (s + 1.) * aa[0][i]

    Key insight: Use scalar expansion to break dependency
        s[0] = d[0]*d[0] if a[0] > 0 else 0
        s[i] = d[i]*d[i] if a[i] > 0 else s[i-1]

    Implementation:
    1. Sequential kernel: Expand scalar s into array s[i]
    2. Parallel kernel: Compute b[i] and e[i] using s[i]
    """
    a = a.contiguous()
    aa = aa.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()

    n_elements = a.shape[0]

    # Allocate temporary array for expanded s
    s_array = torch.empty(n_elements, device=a.device, dtype=a.dtype)

    # Step 1: Expand scalar s into array (sequential)
    grid = (1,)
    s258_expand_s_kernel[grid](
        s_array, a, d,
        n_elements,
    )

    # Step 2: Compute b and e in parallel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_compute_kernel[grid](
        b, e, s_array, c, d, aa[0],
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return b, e
