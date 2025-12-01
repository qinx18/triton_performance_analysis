import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start_idx in range(block_start, min(block_start + BLOCK_SIZE, n_elements - 1), 1):
        idx = start_idx
        mask_idx = idx < n_elements - 1
        
        if mask_idx:
            # Load values
            a_val = tl.load(a_ptr + idx)
            b_val = tl.load(b_ptr + idx)
            c_val = tl.load(c_ptr + idx)
            d_val = tl.load(d_ptr + idx)
            e_val = tl.load(e_ptr + idx)
            
            # Check first condition: if (a[i] >= 0.) goto L20
            if a_val >= 0.0:
                continue  # goto L20 (skip everything)
            
            # Check second condition: if (b[i] >= 0.) goto L30
            if b_val >= 0.0:
                # goto L30: only execute b[i+1] = c[i] + d[i] * e[i]
                b_next_val = c_val + d_val * e_val
                tl.store(b_ptr + idx + 1, b_next_val)
                continue
            
            # Execute both operations (neither condition was true)
            # a[i] += c[i] * d[i]
            new_a_val = a_val + c_val * d_val
            tl.store(a_ptr + idx, new_a_val)
            
            # L30: b[i+1] = c[i] + d[i] * e[i]
            b_next_val = c_val + d_val * e_val
            tl.store(b_ptr + idx + 1, b_next_val)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )