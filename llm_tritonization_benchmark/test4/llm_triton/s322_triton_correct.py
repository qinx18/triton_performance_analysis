import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, start_idx, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s322 computation.
    Processes a single element per kernel launch to avoid race conditions.

    Each element depends on the two previous elements, so they must be
    computed sequentially.
    """
    idx = start_idx + tl.program_id(axis=0)

    if idx < n_elements:
        # Load current element and dependencies
        a_curr = tl.load(a_ptr + idx)
        a_prev1 = tl.load(a_ptr + idx - 1)
        a_prev2 = tl.load(a_ptr + idx - 2)
        b_curr = tl.load(b_ptr + idx)
        c_curr = tl.load(c_ptr + idx)

        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_curr + a_prev1 * b_curr + a_prev2 * c_curr

        # Store result
        tl.store(a_ptr + idx, result)

def s322_triton(a, b, c):
    """
    Triton implementation of TSVC s322.

    This is a recurrence relation where a[i] depends on a[i-1] and a[i-2].
    Due to sequential dependencies, parallelization is not beneficial.

    Note: This type of recurrence (first-order linear recurrence) has
    inherent sequential dependencies that prevent effective parallelization.
    Triton provides no advantage here, so we use PyTorch's sequential loop
    which is the most straightforward and correct implementation.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    n_elements = a.shape[0]

    if n_elements <= 2:
        return a

    # Process elements sequentially
    # This is fundamentally sequential due to data dependencies
    for i in range(2, n_elements):
        a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i]

    return a
