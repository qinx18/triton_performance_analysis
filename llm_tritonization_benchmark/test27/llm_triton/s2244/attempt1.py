import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get offsets for vectorized operations
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process main loop iterations in blocks
    for block_start in range(0, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load values for current block
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # S1: a[i] = b[i] + c[i] (execute for all iterations)
        result_s1 = b_vals + c_vals
        tl.store(a_ptr + current_offsets, result_s1, mask=mask)
    
    # Epilogue: S0 only for last iteration (i = n_elements - 2)
    if tl.program_id(0) == 0:  # Only first work item handles this
        last_i = n_elements - 2
        if last_i >= 0:
            b_val = tl.load(b_ptr + last_i)
            e_val = tl.load(e_ptr + last_i)
            result_s0 = b_val + e_val
            tl.store(a_ptr + last_i + 1, result_s0)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s2244_kernel[(num_blocks,)](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )