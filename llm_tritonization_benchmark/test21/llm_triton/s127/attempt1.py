import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Each thread processes one iteration of the original loop
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Load b[i], c[i], d[i], e[i]
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute values for a[j] and a[j+1]
    # j starts at -1, so j++ makes it 0, 1, 2, 3, ...
    # For iteration i: j = 2*i and j+1 = 2*i+1
    val1 = b_vals + c_vals * d_vals  # a[2*i]
    val2 = b_vals + d_vals * e_vals  # a[2*i+1]
    
    # Store to a[2*i] and a[2*i+1]
    j_offsets_even = 2 * i_offsets
    j_offsets_odd = 2 * i_offsets + 1
    
    tl.store(a_ptr + j_offsets_even, val1, mask=mask)
    tl.store(a_ptr + j_offsets_odd, val2, mask=mask)

def s127_triton(a, b, c, d, e):
    N = a.shape[0]
    n_elements = N // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )