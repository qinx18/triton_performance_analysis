import torch
import triton
import triton.language as tl

@triton.jit
def vas_kernel(
    a_ptr, b_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for scatter operation: a[ip[i]] = b[i]

    Single kernel with loop that processes ALL elements sequentially.
    This ensures correct "last write wins" semantics for duplicate indices.
    """
    # Only one thread does all the work sequentially
    tid = tl.program_id(axis=0)

    if tid == 0:
        # Loop through ALL elements sequentially
        for i in range(n_elements):
            # Load index and value
            ip_val = tl.load(ip_ptr + i)
            b_val = tl.load(b_ptr + i)

            # Scatter: a[ip[i]] = b[i]
            tl.store(a_ptr + ip_val, b_val)

def vas_triton(a, b, ip):
    """
    Triton implementation of TSVC vas - scatter operation.

    Launches ONE kernel where a single thread processes all elements sequentially.
    This ensures correct "last write wins" semantics for duplicate indices.

    Note: This is not faster than CPU for this pattern, but it's correct.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()

    n_elements = b.numel()

    # Launch ONE kernel with a single thread that processes everything
    BLOCK_SIZE = 1
    vas_kernel[(1,)](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a
