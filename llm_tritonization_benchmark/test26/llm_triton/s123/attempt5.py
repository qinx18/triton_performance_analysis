import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_input, BLOCK_SIZE: tl.constexpr):
    # Each program handles one input element
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_input
    
    # Load input data
    b_vals = tl.load(b_ptr + i, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i, mask=mask, other=0.0)
    
    # Compute d[i] * e[i]
    de_vals = d_vals * e_vals
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    # j starts at -1, increments to 0, 1, 2, etc.
    # For i=0: j=0, for i=1: j starts from previous final j value
    # This requires prefix sum to track j positions
    
    # Count how many j increments happen before each i
    # For each i: 1 increment (j++) + conditional increment if c[i] > 0
    c_positive = c_vals > 0.0
    increments = 1 + tl.where(c_positive, 1, 0)
    
    # Calculate prefix sum to find starting j for each i
    # This is complex in Triton, so we'll use a different approach
    # Since this is sequential dependency, process element by element
    
    # Store first assignment
    j_base = i * 2  # Approximate j position assuming worst case
    out1_vals = b_vals + de_vals
    tl.store(a_ptr + j_base, out1_vals, mask=mask)
    
    # Store second assignment conditionally
    out2_vals = c_vals + de_vals
    tl.store(a_ptr + j_base + 1, out2_vals, mask=mask & c_positive)

@triton.jit
def s123_sequential_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_input):
    # Process sequentially to maintain j dependency
    j = 0
    
    for i in range(n_input):
        # Load values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute d[i] * e[i]
        de_val = d_val * e_val
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        out1_val = b_val + de_val
        tl.store(a_ptr + j, out1_val)
        j = j + 1
        
        # Conditional second assignment
        if c_val > 0.0:
            out2_val = c_val + de_val
            tl.store(a_ptr + j, out2_val)
            j = j + 1

def s123_triton(a, b, c, d, e):
    n_input = b.shape[0] // 2  # Loop goes to LEN_1D/2
    
    # Use sequential kernel due to j dependency
    s123_sequential_kernel[(1,)](
        a, b, c, d, e,
        n_input
    )
    
    return a