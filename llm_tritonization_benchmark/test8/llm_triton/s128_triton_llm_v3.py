import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # k = j + 1 = 2*i (since j starts at -1 and increments by 2 each iteration)
        k_offsets = 2 * current_offsets
        k_mask = k_offsets < (2 * n)  # Ensure k is within bounds for array b
        
        # Load values
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        b_k_vals = tl.load(b_ptr + k_offsets, mask=mask & k_mask, other=0.0)
        c_k_vals = tl.load(c_ptr + k_offsets, mask=mask & k_mask, other=0.0)
        
        # a[i] = b[k] - d[i]
        a_vals = b_k_vals - d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # b[k] = a[i] + c[k]
        b_new_vals = a_vals + c_k_vals
        tl.store(b_ptr + k_offsets, b_new_vals, mask=mask & k_mask)

def s128_triton(a, b, c, d):
    n = a.shape[0] // 2
    BLOCK_SIZE = 256
    
    # Create read-only copy of b to handle WAR dependency
    b_copy = b.clone()
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s128_kernel[grid](a, b_copy, c, d, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Copy results back to original b
    b.copy_(b_copy)