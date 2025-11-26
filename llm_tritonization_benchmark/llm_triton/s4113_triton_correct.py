import torch
import triton
import triton.language as tl

@triton.jit
def s4113_kernel(
    a_ptr, b_ptr, c_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for indirect addressing: a[ip[i]] = b[ip[i]] + c[i]

    Single kernel with loop that processes ALL elements sequentially.
    This ensures correct "last write wins" semantics for duplicate indices.
    """
    # Only one thread does all the work sequentially
    tid = tl.program_id(axis=0)

    if tid == 0:
        # Loop through ALL elements sequentially
        for i in range(n_elements):
            # Load index
            ip_val = tl.load(ip_ptr + i)

            # Load values: b[ip[i]] and c[i]
            b_val = tl.load(b_ptr + ip_val)
            c_val = tl.load(c_ptr + i)

            # Compute: b[ip[i]] + c[i]
            result = b_val + c_val

            # Store at indirect location: a[ip[i]] = result
            tl.store(a_ptr + ip_val, result)

def s4113_triton(a, b, c, ip):
    """
    Triton implementation of TSVC s4113 - indirect addressing.

    Launches ONE kernel where a single thread processes all elements sequentially.
    This ensures correct "last write wins" semantics for duplicate indices.

    Note: This is not faster than CPU for this pattern, but it's correct.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    ip = ip.contiguous()

    n_elements = c.numel()

    # Launch ONE kernel with a single thread that processes everything
    BLOCK_SIZE = 1
    s4113_kernel[(1,)](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a
