import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel_else(a_ptr, b_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        c_vals = tl.load(a_ptr + current_offsets, mask=mask)  # c[i] uses a array
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Check condition: c[i] < 0 (else branch)
        else_mask = mask & (c_vals < 0.0)
        
        # Load a[i] for else branch computation
        a_vals = tl.load(a_ptr + current_offsets, mask=else_mask)
        
        # Compute: b[i] = a[i] + d[i] * d[i]
        result = a_vals + d_vals * d_vals
        
        # Store to b array
        tl.store(b_ptr + current_offsets, result, mask=else_mask)

@triton.jit
def s1161_kernel_if(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Check condition: c[i] >= 0 (if branch)
        if_mask = mask & (c_vals >= 0.0)
        
        # Compute: a[i] = c[i] + d[i] * e[i]
        result = c_vals + d_vals * e_vals
        
        # Store to a array
        tl.store(a_ptr + current_offsets, result, mask=if_mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    # Step 1: Process else branch first (c[i] < 0 -> b[i] = a[i] + d[i] * d[i])
    s1161_kernel_else[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        a, b, d, n_elements, BLOCK_SIZE
    )
    
    # Step 2: Process if branch second (c[i] >= 0 -> a[i] = c[i] + d[i] * e[i])
    s1161_kernel_if[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )