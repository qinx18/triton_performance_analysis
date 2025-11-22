import torch
import triton
import triton.language as tl

@triton.jit
def s2251_expand_scalar_kernel(
    s_array_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel 1: Scalar expansion - compute s_array where:
    s_array[0] = 0.0
    s_array[i] = b[i-1] + c[i-1] for i > 0

    This expands the scalar s into an array so we can vectorize the main computation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # For i=0, s_array[0] = 0.0
    # For i>0, s_array[i] = b[i-1] + c[i-1]

    # Load b[i-1] and c[i-1] values (shifted by 1)
    offsets_minus_1 = offsets - 1
    mask_minus_1 = (offsets > 0) & (offsets < n_elements)

    b_prev = tl.load(b_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)
    c_prev = tl.load(c_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)

    # Compute s_array values
    s_vals = b_prev + c_prev

    # For offset 0, override with 0.0
    s_vals = tl.where(offsets == 0, 0.0, s_vals)

    # Store s_array
    tl.store(s_array_ptr + offsets, s_vals, mask=mask)


@triton.jit
def s2251_compute_kernel(
    a_ptr, b_ptr, s_array_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel 2: Vectorized computation using expanded scalar array
    a[i] = s_array[i] * e[i]
    b[i] = a[i] + d[i]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    s_vals = tl.load(s_array_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)

    # Compute a[i] = s_array[i] * e[i]
    a_vals = s_vals * e_vals

    # Compute b[i] = a[i] + d[i]
    b_vals = a_vals + d_vals

    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask)


def s2251_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s2251 using scalar expansion technique.

    Original pattern has sequential dependency:
        s = 0.0
        for i in range(N):
            a[i] = s * e[i]        # Uses s from previous iteration
            s = b[i] + c[i]        # Update s for next iteration
            b[i] = a[i] + d[i]

    Scalar expansion breaks the dependency:
    1. Expand scalar s into array s_array where:
       - s_array[0] = 0.0
       - s_array[i] = b[i-1] + c[i-1] for i > 0

    2. Vectorize the computation:
       - a[i] = s_array[i] * e[i]
       - b[i] = a[i] + d[i]

    This allows full parallelization!
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()

    n_elements = a.shape[0]

    # Create temporary array for scalar expansion
    s_array = torch.empty_like(a)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Step 1: Expand scalar s into array s_array
    s2251_expand_scalar_kernel[grid](
        s_array, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2: Vectorized computation using s_array
    s2251_compute_kernel[grid](
        a, b, s_array, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a, b
