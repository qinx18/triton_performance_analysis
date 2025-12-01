import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a = tl.load(a_ptr + block_start + offsets, mask=mask)
    b = tl.load(b_ptr + block_start + offsets, mask=mask)
    c = tl.load(c_ptr + block_start + offsets, mask=mask)
    d = tl.load(d_ptr + block_start + offsets, mask=mask)
    e = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Control flow logic
    cond1 = a > 0.0  # if (a[i] > 0.) goto L20
    
    # Path when a[i] <= 0 (not goto L20)
    b_new = -b + d * d
    cond2 = b_new <= a  # if (b[i] <= a[i]) goto L30
    
    # Update c based on conditions
    # When a[i] > 0: c[i] = -c[i] + e[i] * e[i] (L20)
    c_l20 = -c + e * e
    
    # When a[i] <= 0 and b[i] > a[i]: c[i] += d[i] * e[i]
    c_middle = c + d * e
    
    # Select appropriate c value
    c_final = tl.where(cond1, c_l20, tl.where(cond2, c, c_middle))
    
    # Update b only when a[i] <= 0
    b_final = tl.where(cond1, b, b_new)
    
    # L30: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d
    
    # Store results
    tl.store(a_ptr + block_start + offsets, a_final, mask=mask)
    tl.store(b_ptr + block_start + offsets, b_final, mask=mask)
    tl.store(c_ptr + block_start + offsets, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )