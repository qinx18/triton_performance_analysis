import torch
import triton
import triton.language as tl

@triton.jit
def s2111_diagonal_kernel(
    aa_ptr,
    M, N,
    diag_idx,
    start_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to process one diagonal of the matrix.
    Elements on the same diagonal can be computed in parallel.

    For diagonal d:
    - Elements are at positions (j, i) where i + j = d + 2
    - Valid range: j in [1, min(d+1, M-1)], i = d + 2 - j
    """
    # Get thread ID within this diagonal
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Calculate actual index within diagonal (accounting for offset)
    actual_pid = pid + start_offset

    # Calculate row and column indices for this thread
    # For diagonal d, we have elements at positions where i + j = d + 2
    # In general: j = actual_pid + 1, i = d + 1 - actual_pid

    j = actual_pid + 1
    i = diag_idx + 1 - actual_pid

    # Create mask for valid threads
    mask = (j < M) & (i < N) & (i >= 1)

    # Calculate memory offsets
    curr_offset = j * N + i
    left_offset = j * N + (i - 1)
    up_offset = (j - 1) * N + i

    # Load values with mask
    left_val = tl.load(aa_ptr + left_offset, mask=mask)
    up_val = tl.load(aa_ptr + up_offset, mask=mask)

    # Compute and store
    new_val = (left_val + up_val) / 1.9
    tl.store(aa_ptr + curr_offset, new_val, mask=mask)

def s2111_triton(aa):
    """
    Triton implementation of TSVC s2111 using diagonal wavefront processing.

    Algorithm:
    - Process matrix along diagonals
    - Elements on same diagonal have no dependencies, can be parallel
    - Diagonals are processed sequentially

    For a matrix starting at (1,1), diagonals are:
    - Diagonal 0: (1,1)
    - Diagonal 1: (1,2), (2,1)
    - Diagonal 2: (1,3), (2,2), (3,1)
    - ...
    """
    aa = aa.contiguous()
    M, N = aa.shape

    # Number of diagonals to process
    # Maximum diagonal index is (M-1) + (N-1) - 2
    max_diag = M + N - 4

    BLOCK_SIZE = 256

    # Process each diagonal sequentially
    for diag in range(max_diag + 1):
        # Calculate start and end pid for this diagonal
        # For diagonal d, we have i + j = d + 2
        # Start: when i < N, we need d + 1 - pid < N, so pid > d + 1 - N
        # Also j >= 1, so pid >= 0
        start_pid = max(0, diag + 2 - N)

        # End: when j < M, we need pid + 1 < M, so pid < M - 1
        # Also i >= 1, so d + 1 - pid >= 1, so pid <= d
        end_pid = min(diag, M - 2)

        num_elements = end_pid - start_pid + 1

        if num_elements <= 0:
            continue

        # Launch kernel for this diagonal
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        s2111_diagonal_kernel[grid](
            aa,
            M, N,
            diag,
            start_pid,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return aa
