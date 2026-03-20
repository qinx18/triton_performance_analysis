import triton
import triton.language as tl

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    for t in range(TSTEPS):
        # First phase: A -> B
        for block_start in range(0, (N-2) * (N-2) * (N-2), BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (N-2) * (N-2) * (N-2)
            
            # Convert linear index to 3D coordinates (1-based indexing)
            i = offsets // ((N-2) * (N-2)) + 1
            j = (offsets % ((N-2) * (N-2))) // (N-2) + 1
            k = offsets % (N-2) + 1
            
            # Calculate 3D array indices
            center_idx = i * N * N + j * N + k
            i_plus_idx = (i + 1) * N * N + j * N + k
            i_minus_idx = (i - 1) * N * N + j * N + k
            j_plus_idx = i * N * N + (j + 1) * N + k
            j_minus_idx = i * N * N + (j - 1) * N + k
            k_plus_idx = i * N * N + j * N + (k + 1)
            k_minus_idx = i * N * N + j * N + (k - 1)
            
            # Load values
            a_center = tl.load(A_ptr + center_idx, mask=mask)
            a_i_plus = tl.load(A_ptr + i_plus_idx, mask=mask)
            a_i_minus = tl.load(A_ptr + i_minus_idx, mask=mask)
            a_j_plus = tl.load(A_ptr + j_plus_idx, mask=mask)
            a_j_minus = tl.load(A_ptr + j_minus_idx, mask=mask)
            a_k_plus = tl.load(A_ptr + k_plus_idx, mask=mask)
            a_k_minus = tl.load(A_ptr + k_minus_idx, mask=mask)
            
            # Compute heat equation
            result = (0.125 * (a_i_plus - 2.0 * a_center + a_i_minus) +
                     0.125 * (a_j_plus - 2.0 * a_center + a_j_minus) +
                     0.125 * (a_k_plus - 2.0 * a_center + a_k_minus) +
                     a_center)
            
            # Store to B
            tl.store(B_ptr + center_idx, result, mask=mask)
        
        # Second phase: B -> A
        for block_start in range(0, (N-2) * (N-2) * (N-2), BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (N-2) * (N-2) * (N-2)
            
            # Convert linear index to 3D coordinates (1-based indexing)
            i = offsets // ((N-2) * (N-2)) + 1
            j = (offsets % ((N-2) * (N-2))) // (N-2) + 1
            k = offsets % (N-2) + 1
            
            # Calculate 3D array indices
            center_idx = i * N * N + j * N + k
            i_plus_idx = (i + 1) * N * N + j * N + k
            i_minus_idx = (i - 1) * N * N + j * N + k
            j_plus_idx = i * N * N + (j + 1) * N + k
            j_minus_idx = i * N * N + (j - 1) * N + k
            k_plus_idx = i * N * N + j * N + (k + 1)
            k_minus_idx = i * N * N + j * N + (k - 1)
            
            # Load values
            b_center = tl.load(B_ptr + center_idx, mask=mask)
            b_i_plus = tl.load(B_ptr + i_plus_idx, mask=mask)
            b_i_minus = tl.load(B_ptr + i_minus_idx, mask=mask)
            b_j_plus = tl.load(B_ptr + j_plus_idx, mask=mask)
            b_j_minus = tl.load(B_ptr + j_minus_idx, mask=mask)
            b_k_plus = tl.load(B_ptr + k_plus_idx, mask=mask)
            b_k_minus = tl.load(B_ptr + k_minus_idx, mask=mask)
            
            # Compute heat equation
            result = (0.125 * (b_i_plus - 2.0 * b_center + b_i_minus) +
                     0.125 * (b_j_plus - 2.0 * b_center + b_j_minus) +
                     0.125 * (b_k_plus - 2.0 * b_center + b_k_minus) +
                     b_center)
            
            # Store to A
            tl.store(A_ptr + center_idx, result, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 256
    heat_3d_kernel[(1,)](A, B, N, TSTEPS, BLOCK_SIZE)