import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        
        # Check if indices are in bounds and follow the stride pattern
        valid = current_offsets < N
        
        # Check if indices match the pattern: i = n1-1 + k*n3 for some k >= 0
        # This means (current_offsets - (n1-1)) % n3 == 0 and current_offsets >= n1-1
        pattern_match = ((current_offsets >= (n1-1)) & 
                        ((current_offsets - (n1-1)) % n3 == 0))
        
        mask = valid & pattern_match
        
        # Load values where mask is true
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Perform computation
        result = a_vals + b_vals
        
        # Store back
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    s172_kernel[(1,)](a, b, n1, n3, N, BLOCK_SIZE)