import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a, b, c, a_orig, n, BLOCK_SIZE: tl.constexpr):
    """Unified kernel handling both phases based on index position"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load indices for reverse access
    reverse_offsets = n - 1 - offsets
    
    # Determine which array to read from based on crossing threshold
    threshold = n // 2
    
    # For indices < threshold: use original array (Phase 1)
    # For indices >= threshold: use updated array (Phase 2)
    use_orig = offsets < threshold
    
    # Load from appropriate source
    a_vals = tl.where(use_orig,
                      tl.load(a_orig + reverse_offsets, mask=mask),
                      tl.load(a + reverse_offsets, mask=mask))
    
    b_vals = tl.load(b + offsets, mask=mask)
    c_vals = tl.load(c + offsets, mask=mask)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x_vals = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a + offsets, x_vals - 1.0, mask=mask)
    tl.store(b + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Create copy of original array for phase 1 reads
    a_orig = a.clone()
    
    # Phase 1: Process indices 0 to threshold-1
    # These read from original values at high indices
    if threshold > 0:
        grid = (triton.cdiv(threshold, BLOCK_SIZE),)
        s281_kernel[grid](a, b, c, a_orig, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Process indices threshold to n-1  
    # These read from updated values at low indices
    remaining = n - threshold
    if remaining > 0:
        # Shift arrays to process remaining elements
        a_phase2 = a[threshold:]
        b_phase2 = b[threshold:]
        c_phase2 = c[threshold:]
        
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        
        # Create adjusted kernel for phase 2
        @triton.jit
        def s281_phase2_kernel(a_out, b_out, c_in, a_full, n_full, offset, n_phase, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_phase
            
            # Global indices for reverse lookup
            global_offsets = offsets + offset
            reverse_offsets = n_full - 1 - global_offsets
            
            # Load values
            a_vals = tl.load(a_full + reverse_offsets, mask=mask)
            b_vals = tl.load(c_in + offsets, mask=mask)  # b[i] from phase2 slice
            c_vals = tl.load(c_in + offsets, mask=mask)  # Actually c[i]
            
            # Need to load correct b and c values
            b_vals = tl.load(b_out + offsets, mask=mask)  # Current b values
            c_vals = tl.load(c_in + offsets, mask=mask)   # Current c values
            
            x_vals = a_vals + b_vals * c_vals
            
            tl.store(a_out + offsets, x_vals - 1.0, mask=mask)
            tl.store(b_out + offsets, x_vals, mask=mask)
        
        s281_phase2_kernel[grid](a_phase2, b_phase2, c_phase2, a, n, threshold, remaining, BLOCK_SIZE=BLOCK_SIZE)