import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(
    aa_ptr,
    bb_ptr,
    M, N,
    i_val,
    stride_aa_0, stride_aa_1,
    stride_bb_0, stride_bb_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s119 - processes one row i at a time.
    Parallelizes over j dimension (columns).

    For fixed i, computes: aa[i, j] = aa[i-1, j-1] + bb[i, j] for all j >= 1
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Calculate j offsets (j starts from 1)
    j_offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1

    # Mask for valid j values (1 <= j < N)
    mask = j_offsets < N

    # Calculate memory addresses for aa[i, j]
    aa_curr_addrs = aa_ptr + i_val * stride_aa_0 + j_offsets * stride_aa_1

    # Calculate addresses for aa[i-1, j-1]
    aa_prev_addrs = aa_ptr + (i_val - 1) * stride_aa_0 + (j_offsets - 1) * stride_aa_1

    # Calculate addresses for bb[i, j]
    bb_curr_addrs = bb_ptr + i_val * stride_bb_0 + j_offsets * stride_bb_1

    # Load values
    aa_prev = tl.load(aa_prev_addrs, mask=mask, other=0.0)
    bb_curr = tl.load(bb_curr_addrs, mask=mask, other=0.0)

    # Compute result: aa[i, j] = aa[i-1, j-1] + bb[i, j]
    result = aa_prev + bb_curr

    # Store result
    tl.store(aa_curr_addrs, result, mask=mask)


def s119_triton(aa, bb):
    """
    Triton implementation of TSVC s119 - 2D array diagonal dependency.

    Original C code:
    for (int i = 1; i < LEN_2D; i++) {
        for (int j = 1; j < LEN_2D; j++) {
            aa[i][j] = aa[i-1][j-1] + bb[i][j];
        }
    }

    FIXED: Launches kernels SEQUENTIALLY for each i to respect diagonal dependencies.
    Each kernel parallelizes over j values (columns).
    """
    aa = aa.contiguous()
    bb = bb.contiguous()

    M, N = aa.shape

    if M <= 1 or N <= 1:
        return aa

    BLOCK_SIZE = 256

    # Process each row i SEQUENTIALLY (i from 1 to M-1)
    for i in range(1, M):
        # Number of j values to process (j from 1 to N-1)
        num_j = N - 1

        # Calculate grid size for this row
        grid_size = triton.cdiv(num_j, BLOCK_SIZE)

        # Launch kernel for this specific row i
        s119_kernel[(grid_size,)](
            aa_ptr=aa,
            bb_ptr=bb,
            M=M,
            N=N,
            i_val=i,
            stride_aa_0=aa.stride(0),
            stride_aa_1=aa.stride(1),
            stride_bb_0=bb.stride(0),
            stride_bb_1=bb.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return aa
