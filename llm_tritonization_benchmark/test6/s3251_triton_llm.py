import torch
import triton
import triton.language as tl

@triton.jit
def s3251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s3251 computation.
    Processes elements in parallel blocks with proper dependency handling.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_elements - 1)
    
    # Load input data for current block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)  # Need current a[i] for d[i] computation
    
    # Compute operations
    # a[i+1] = b[i] + c[i] - store at offset+1
    a_new = b_vals + c_vals
    tl.store(a_ptr + offsets + 1, a_new, mask=mask)
    
    # b[i] = c[i] * e[i]
    b_new = c_vals * e_vals
    tl.store(b_ptr + offsets, b_new, mask=mask)
    
    # d[i] = a[i] * e[i] - uses original a[i] values
    d_new = a_vals * e_vals
    tl.store(d_ptr + offsets, d_new, mask=mask)

def s3251_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s3251.
    Optimized GPU kernel with block-parallel processing.
    """
    a = a.contiguous().clone()
    b = b.contiguous().clone()
    d = d.contiguous().clone()
    
    LEN_1D = len(a)
    n_elements = LEN_1D - 1
    
    # Use power-of-2 block size for memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, d