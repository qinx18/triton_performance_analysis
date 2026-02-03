import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, a_orig_ptr, n, phase, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    if phase == 1:
        # Phase 1: i = 0 to n//2 - 1
        indices = pid * BLOCK_SIZE + offsets
        mask = indices < n // 2
        
        # Load b[i] and c[i]
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
        
        # Load a_orig[n-1-i] (original values)
        reverse_indices = n - 1 - indices
        a_reverse_vals = tl.load(a_orig_ptr + reverse_indices, mask=mask, other=0.0)
        
        # Compute x = a[n-1-i] + b[i] * c[i]
        x_vals = a_reverse_vals + b_vals * c_vals
        
        # Store a[i] = x - 1.0 and b[i] = x
        tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
        tl.store(b_ptr + indices, x_vals, mask=mask)
    
    else:
        # Phase 2: i = n//2 to n-1
        indices = pid * BLOCK_SIZE + offsets + n // 2
        mask = indices < n
        
        # Load b[i] and c[i]
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
        
        # Load a[n-1-i] (updated values from phase 1)
        reverse_indices = n - 1 - indices
        a_reverse_vals = tl.load(a_ptr + reverse_indices, mask=mask, other=0.0)
        
        # Compute x = a[n-1-i] + b[i] * c[i]
        x_vals = a_reverse_vals + b_vals * c_vals
        
        # Store a[i] = x - 1.0 and b[i] = x
        tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
        tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    
    # Create copy of original a values for phase 1
    a_orig = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: process first half (i = 0 to n//2 - 1)
    phase1_size = n // 2
    grid1 = (triton.cdiv(phase1_size, BLOCK_SIZE),)
    s281_kernel[grid1](a, b, c, a_orig, n, 1, BLOCK_SIZE)
    
    # Phase 2: process second half (i = n//2 to n-1)
    phase2_size = n - n // 2
    grid2 = (triton.cdiv(phase2_size, BLOCK_SIZE),)
    s281_kernel[grid2](a, b, c, a_orig, n, 2, BLOCK_SIZE)