import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    len_1d,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1111 computation.
    Computes: a[2*i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    """
    # Get program ID and compute element indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements (i < len_1d // 2)
    mask = offsets < (len_1d // 2)
    
    # Load input values with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute the expression: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    # Factoring for efficiency: b[i] * (c[i] + 2*d[i]) + c[i] * (c[i] + d[i])
    result = b_vals * (c_vals + 2.0 * d_vals) + c_vals * (c_vals + d_vals)
    
    # Store results at a[2*i] positions
    output_offsets = 2 * offsets
    tl.store(a_ptr + output_offsets, result, mask=mask)

def s1111_triton(a, b, c, d, iterations):
    """
    Triton implementation of TSVC s1111 function.
    
    Optimizations:
    - Vectorized computation using Triton blocks
    - Factored arithmetic expression to reduce operations
    - Coalesced memory access patterns
    - Efficient masking for edge cases
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    len_1d = a.size(0)
    n_elements = len_1d // 2
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Execute iterations
    for nl in range(2 * iterations):
        s1111_kernel[grid](
            a, b, c, d,
            len_1d,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return a