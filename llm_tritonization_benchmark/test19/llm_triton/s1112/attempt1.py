import triton
import triton.language as tl

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calculate block start position from the end
    block_id = tl.program_id(0)
    total_blocks = tl.num_programs(0)
    
    # Start from the end and work backwards
    block_start = n - (block_id + 1) * BLOCK_SIZE
    
    # Generate offsets within the block
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices (must be >= 0 and < n)
    mask = (indices >= 0) & (indices < n)
    
    # Load data
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute: a[i] = b[i] + 1.0
    a_vals = b_vals + 1.0
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s1112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    grid = (grid_size,)
    
    # Launch kernel
    s1112_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)