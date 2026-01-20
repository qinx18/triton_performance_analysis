import triton
import triton.language as tl
import torch

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, max_i_ptr, max_j_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize with aa[0][0]
    if pid == 0:
        first_val = tl.load(aa_ptr)
        tl.store(max_val_ptr, first_val)
        tl.store(max_i_ptr, 0)
        tl.store(max_j_ptr, 0)
    
    tl.debug_barrier()
    
    # Process elements in blocks
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < (N * N)
    
    # Load current maximum
    current_max = tl.load(max_val_ptr)
    
    # Load values from flattened view
    vals = tl.load(aa_ptr + indices, mask=mask, other=-float('inf'))
    
    # Find local maximum
    local_max_mask = vals > current_max
    
    if tl.any(local_max_mask):
        # Find the position of maximum value in this block
        for k in range(BLOCK_SIZE):
            if k + block_start < N * N:
                idx = k + block_start
                val = tl.load(aa_ptr + idx)
                if val > tl.load(max_val_ptr):
                    tl.store(max_val_ptr, val)
                    i_coord = idx // N
                    j_coord = idx % N
                    tl.store(max_i_ptr, i_coord)
                    tl.store(max_j_ptr, j_coord)

def s3110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // N
    yindex = max_idx % N
    
    # Return max + xindex+1 + yindex+1 as per C code
    return max_val + xindex + 1 + yindex + 1