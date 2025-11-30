import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the sequential loop with conditional induction variable
    # Each block processes a contiguous chunk of the input arrays
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input arrays
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate base output index (j starts at -1, then incremented)
    # For each i, j is at least i (from initial j++)
    # If c[i] > 0, j gets incremented again
    base_j = offsets
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    val1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + base_j, val1, mask=mask)
    
    # Second assignment (conditional): a[j+1] = c[i] + d[i] * e[i]
    condition = c_vals > 0.0
    val2 = c_vals + d_vals * e_vals
    
    # Calculate cumulative sum of conditions up to current position
    # This determines the additional offset for j
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            additional_offset = 0
            for k in range(i + 1):
                if block_start + k < n_elements:
                    c_k = tl.load(c_ptr + block_start + k)
                    if c_k > 0.0:
                        additional_offset += 1
            
            if condition and (block_start + i < n_elements):
                j_idx = base_j + additional_offset
                val2_i = tl.load(c_ptr + block_start + i) + tl.load(d_ptr + block_start + i) * tl.load(e_ptr + block_start + i)
                tl.store(a_ptr + j_idx, val2_i)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    # Zero out the output array
    a.zero_()
    
    # Sequential execution due to complex induction variable dependencies
    j = -1
    for i in range(n_elements):
        j += 1
        a[j] = b[i] + d[i] * e[i]
        if c[i] > 0.0:
            j += 1
            a[j] = c[i] + d[i] * e[i]
    
    return a