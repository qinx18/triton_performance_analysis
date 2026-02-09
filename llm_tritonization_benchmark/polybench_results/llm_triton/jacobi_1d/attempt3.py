import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, A_copy_ptr, B_copy_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # Copy current A to A_copy for reading
        for block_start in range(0, N, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N
            
            A_vals = tl.load(A_ptr + current_offsets, mask=mask, other=0.0)
            tl.store(A_copy_ptr + current_offsets, A_vals, mask=mask)
        
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        for block_start in range(1, N-1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N-1
            
            A_left = tl.load(A_copy_ptr + current_offsets - 1, mask=mask, other=0.0)
            A_center = tl.load(A_copy_ptr + current_offsets, mask=mask, other=0.0)
            A_right = tl.load(A_copy_ptr + current_offsets + 1, mask=mask, other=0.0)
            
            B_new = 0.33333 * (A_left + A_center + A_right)
            tl.store(B_ptr + current_offsets, B_new, mask=mask)
        
        # Copy current B to B_copy for reading
        for block_start in range(0, N, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N
            
            B_vals = tl.load(B_ptr + current_offsets, mask=mask, other=0.0)
            tl.store(B_copy_ptr + current_offsets, B_vals, mask=mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        for block_start in range(1, N-1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N-1
            
            B_left = tl.load(B_copy_ptr + current_offsets - 1, mask=mask, other=0.0)
            B_center = tl.load(B_copy_ptr + current_offsets, mask=mask, other=0.0)
            B_right = tl.load(B_copy_ptr + current_offsets + 1, mask=mask, other=0.0)
            
            A_new = 0.33333 * (B_left + B_center + B_right)
            tl.store(A_ptr + current_offsets, A_new, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    
    A_copy = torch.empty_like(A)
    B_copy = torch.empty_like(B)
    
    grid = (1,)
    
    jacobi_1d_kernel[grid](
        A, B, A_copy, B_copy, N, TSTEPS,
        BLOCK_SIZE=BLOCK_SIZE
    )