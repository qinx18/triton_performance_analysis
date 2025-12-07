import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Load input arrays
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Calculate the base j index for each i
    # j starts at -1, then j++ before first store, so j = i
    # For conditional store, j++ again, so j = i + 1 + (i * number_of_previous_conditional_stores)
    
    # We need to compute cumulative sum of conditions for each element
    # For element i, j_base = i + sum(c[0:i] > 0)
    j_base = i_offsets
    
    # Add cumulative count of positive c values up to current position
    for k in range(BLOCK_SIZE):
        if block_start + k < n_elements:
            count = 0
            for prev in range(k):
                if block_start + prev < n_elements:
                    prev_idx = prev
                    c_prev = tl.load(c_ptr + block_start + prev_idx)
                    if c_prev > 0.0:
                        count += 1
            if k < BLOCK_SIZE:
                current_j = i_offsets + count
                # First store: a[j] = b[i] + d[i] * e[i] where j = current_j
                if block_start + k < n_elements and k < BLOCK_SIZE:
                    result1 = b_vals + d_vals * e_vals
                    elem_mask = (offsets == k) & mask
                    tl.store(a_ptr + current_j, result1, mask=elem_mask)
                    
                    # Second store if c[i] > 0: a[j+1] = c[i] + d[i] * e[i]
                    c_positive = c_vals > 0.0
                    result2 = c_vals + d_vals * e_vals
                    store_mask = elem_mask & c_positive
                    tl.store(a_ptr + current_j + 1, result2, mask=store_mask)

@triton.jit
def s123_kernel_sequential(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
        
    j = 0
    for i in range(n_elements):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # First store: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        j += 1
        
        # Second store if c[i] > 0: a[j] = c[i] + d[i] * e[i]
        if c_val > 0.0:
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)
            j += 1

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    # Use sequential kernel due to complex dependency pattern
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s123_kernel_sequential[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )