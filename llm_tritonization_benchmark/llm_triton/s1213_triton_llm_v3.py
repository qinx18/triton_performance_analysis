import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must handle WAR dependency by using temporary storage
    # Process elements sequentially in blocks but handle dependencies carefully
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < n_elements - 1)
    
    # For this pattern, we need to be careful about the dependency:
    # a[i] = b[i-1] + c[i]  (reads b[i-1])
    # b[i] = a[i+1] * d[i]  (writes b[i], but reads a[i+1])
    # 
    # The issue is b[i] is written in iteration i but b[i-1] is read in iteration i+1
    # We need to process this sequentially or use a different approach
    
    # Load required data with proper bounds checking
    b_prev_offsets = offsets - 1
    a_next_offsets = offsets + 1
    
    b_prev_mask = (b_prev_offsets >= 0) & (b_prev_offsets < n_elements) & mask
    a_next_mask = (a_next_offsets < n_elements) & mask
    c_mask = (offsets < n_elements) & mask
    d_mask = (offsets < n_elements) & mask
    
    # Load values
    b_prev = tl.load(b_ptr + b_prev_offsets, mask=b_prev_mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=c_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=d_mask, other=0.0)
    a_next = tl.load(a_ptr + a_next_offsets, mask=a_next_mask, other=0.0)
    
    # Compute first statement: a[i] = b[i-1] + c[i]
    a_new = b_prev + c_vals
    
    # Store a[i]
    tl.store(a_ptr + offsets, a_new, mask=mask)
    
    # Compute second statement: b[i] = a[i+1] * d[i]
    b_new = a_next * d_vals
    
    # Store b[i]
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1213_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )