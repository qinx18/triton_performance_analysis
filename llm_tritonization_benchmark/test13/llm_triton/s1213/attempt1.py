import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process strips sequentially due to loop-carried dependency
    for strip_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = strip_start + offsets
        mask = current_offsets < n_elements
        
        # Adjust indices to match original loop range [1, LEN_1D-1)
        i_indices = current_offsets + 1
        
        # Load values for a[i] = b[i-1] + c[i]
        b_prev = tl.load(b_ptr + i_indices - 1, mask=mask)
        c_vals = tl.load(c_ptr + i_indices, mask=mask)
        a_result = b_prev + c_vals
        tl.store(a_ptr + i_indices, a_result, mask=mask)
        
        # Load values for b[i] = a[i+1] * d[i]
        a_next = tl.load(a_copy_ptr + i_indices + 1, mask=mask)
        d_vals = tl.load(d_ptr + i_indices, mask=mask)
        b_result = a_next * d_vals
        tl.store(b_ptr + i_indices, b_result, mask=mask)

def s1213_triton(a, b, c, d):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 2  # Loop range is [1, LEN_1D-1), so n_elements = LEN_1D-2
    
    if n_elements <= 0:
        return
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 1  # Must be 1 due to loop-carried dependency
    
    s1213_kernel[(1,)](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )