import torch
import triton
import triton.language as tl

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid elements (starting from index 1)
    mask = (idx < n_elements) & (idx >= 0)
    
    # Load values
    b_prev = tl.load(b_ptr + idx, mask=mask)  # b[i-1] where idx represents i-1
    c_vals = tl.load(c_ptr + idx + 1, mask=mask)  # c[i] where i = idx + 1
    d_vals = tl.load(d_ptr + idx + 1, mask=mask)  # d[i]
    e_vals = tl.load(e_ptr + idx + 1, mask=mask)  # e[i]
    
    # Compute a[i] = b[i-1] + c[i] * d[i]
    a_new = b_prev + c_vals * d_vals
    
    # Compute b[i] = a[i] + c[i] * e[i]
    b_new = a_new + c_vals * e_vals
    
    # Store results
    tl.store(a_ptr + idx + 1, a_new, mask=mask)
    tl.store(b_ptr + idx + 1, b_new, mask=mask)

def s323_triton(a, b, c, d, e):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1  # We process from index 1 to LEN_1D-1
    
    # Since we have loop-carried dependency in b array, process sequentially
    # Each element must be processed one at a time
    BLOCK_SIZE = 1
    
    for i in range(n_elements):
        s323_kernel[(1,)](
            a, b, c, d, e,
            1,  # Process one element at a time
            BLOCK_SIZE=BLOCK_SIZE
        )
        # Move to next element by adjusting pointers
        if i < n_elements - 1:
            a_offset = a[i:]
            b_offset = b[i:]
            c_offset = c[i:]
            d_offset = d[i:]
            e_offset = e[i:]
            
            s323_kernel[(1,)](
                a_offset, b_offset, c_offset, d_offset, e_offset,
                1,
                BLOCK_SIZE=BLOCK_SIZE
            )