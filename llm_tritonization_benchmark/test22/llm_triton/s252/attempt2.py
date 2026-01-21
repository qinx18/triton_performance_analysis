import triton
import triton.language as tl

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is a sequential scan operation that cannot be parallelized
    # Each element depends on the previous computation
    # We need to process the entire array sequentially in a single thread
    
    # Only process if this is thread 0
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize t = 0
    t = 0.0
    
    # Process in blocks to handle memory efficiently
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load b and c values for this block
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            actual_idx = block_start + i
            if actual_idx >= n:
                return
                
            # Extract scalar values for sequential computation
            b_val = tl.load(b_ptr + actual_idx)
            c_val = tl.load(c_ptr + actual_idx)
            
            # s = b[i] * c[i]
            s = b_val * c_val
            
            # a[i] = s + t
            a_val = s + t
            tl.store(a_ptr + actual_idx, a_val)
            
            # t = s
            t = s

def s252_triton(a, b, c):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Since this is sequential, we only need one thread block
    grid = (1,)
    
    s252_kernel[grid](
        a, b, c, n,
        BLOCK_SIZE=BLOCK_SIZE
    )