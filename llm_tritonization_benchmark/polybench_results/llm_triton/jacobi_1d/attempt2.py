import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    
    for t in range(TSTEPS):
        tl.debug_barrier()
        
        # Phase 1: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1]) for i in [1, N-2]
        for block_start in range(1, N - 1, BLOCK):
            current_offsets = block_start + offsets
            current_mask = current_offsets < (N - 1)
            
            # Load A[i-1], A[i], A[i+1]
            A_left = tl.load(A_ptr + current_offsets - 1, mask=current_mask)
            A_center = tl.load(A_ptr + current_offsets, mask=current_mask)
            A_right = tl.load(A_ptr + current_offsets + 1, mask=current_mask)
            
            # Compute B[i]
            B_vals = 0.33333 * (A_left + A_center + A_right)
            
            # Store B[i]
            tl.store(B_ptr + current_offsets, B_vals, mask=current_mask)
        
        tl.debug_barrier()
        
        # Phase 2: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1]) for i in [1, N-2]
        for block_start in range(1, N - 1, BLOCK):
            current_offsets = block_start + offsets
            current_mask = current_offsets < (N - 1)
            
            # Load B[i-1], B[i], B[i+1]
            B_left = tl.load(B_ptr + current_offsets - 1, mask=current_mask)
            B_center = tl.load(B_ptr + current_offsets, mask=current_mask)
            B_right = tl.load(B_ptr + current_offsets + 1, mask=current_mask)
            
            # Compute A[i]
            A_vals = 0.33333 * (B_left + B_center + B_right)
            
            # Store A[i]
            tl.store(A_ptr + current_offsets, A_vals, mask=current_mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK = triton.next_power_of_2(N)
    
    jacobi_1d_kernel[(1,)](
        A, B, N, TSTEPS, BLOCK
    )