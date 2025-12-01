import torch
import triton
import triton.language as tl

@triton.jit
def s491_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for indirect assignment: a[ip[i]] = b[i] + c[i] * d[i]

    Single kernel with loop that processes ALL elements sequentially.
    This ensures correct "last write wins" semantics for duplicate indices.
    """
    # Only one thread does all the work sequentially
    tid = tl.program_id(axis=0)

    if tid == 0:
        # Loop through ALL elements sequentially
        for i in range(n_elements):
            # Load scalar values for this element
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            ip_val = tl.load(ip_ptr + i)

            # Compute: b[i] + c[i] * d[i]
            result = b_val + c_val * d_val

            # Store at indirect location: a[ip[i]] = result
            tl.store(a_ptr + ip_val, result)

def s491_triton(a, b, c, d, ip):
    """
    Triton implementation of TSVC s491 - indirect assignment (scatter).

    Launches ONE kernel where a single thread processes all elements sequentially.
    This ensures correct "last write wins" semantics for duplicate indices.

    Note: This is not faster than CPU for this pattern, but it's correct.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    ip = ip.contiguous()

    n_elements = b.numel()

    # Launch ONE kernel with a single thread that processes everything
    BLOCK_SIZE = 1
    s491_kernel[(1,)](
        a, b, c, d, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a
