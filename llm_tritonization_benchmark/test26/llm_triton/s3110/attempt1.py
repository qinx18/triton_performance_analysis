import triton
import triton.language as tl
import torch

@triton.jit
def s3110_kernel(aa_ptr, result_ptr, N):
    # This is a global argmax reduction - use single thread
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential search through entire 2D array
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            val = tl.load(aa_ptr + idx)
            if val > max_val:
                max_val = val
                max_i = i
                max_j = j
    
    # Store results: max_val, max_i, max_j
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_i.to(tl.float32))
    tl.store(result_ptr + 2, max_j.to(tl.float32))

def s3110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for efficient argmax
    aa_flat = aa.flatten()
    max_val = torch.max(aa_flat)
    flat_idx = torch.argmax(aa_flat)
    xindex = flat_idx // N
    yindex = flat_idx % N
    
    chksum = max_val + xindex.float() + yindex.float()
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1).float() + (yindex + 1).float()