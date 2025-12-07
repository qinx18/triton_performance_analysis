import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # S0: a[i] = b[i] + c[i] * d[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        result_0 = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, result_0, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        result_1 = c_vals + b_vals
        tl.store(b_ptr + current_offsets, result_1, mask=mask)
    
    # Epilogue - S2 only at last iteration (i = n_elements - 1)
    if n_elements > 0:
        i = n_elements - 1
        if tl.program_id(0) == 0:  # Only first thread block handles epilogue
            # S2: a[i+1] = b[i] + a[i+1] * d[i]
            if i + 1 < n_elements + 1:  # Safety check for array bounds
                b_val = tl.load(b_ptr + i)
                a_val = tl.load(a_copy_ptr + i + 1)
                d_val = tl.load(d_ptr + i)
                
                result_2 = b_val + a_val * d_val
                tl.store(a_ptr + i + 1, result_2)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )