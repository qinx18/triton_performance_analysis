import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(
    a_ptr,
    bb_ptr,
    n,
    i_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s118 computation.
    Processes one specific value of i, parallelizing over j.

    For i_val, computes: a[i] += sum(bb[j, i] * a[i-j-1] for j in 0..i-1)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Calculate j offsets for this block
    j_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid j values (j < i_val)
    mask = j_offsets < i_val

    # Load bb[j, i_val] values
    bb_offsets = j_offsets * n + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)

    # Load a[i_val - j - 1] values
    a_offsets = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)

    # Compute partial products: bb[j, i] * a[i-j-1]
    products = bb_vals * a_vals

    # Sum the products
    result = tl.sum(products, axis=0)

    # Atomically add to a[i_val]
    tl.atomic_add(a_ptr + i_val, result)


def s118_triton(a, bb):
    """
    Triton implementation of TSVC s118 function.

    Original C code:
    for (int i = 1; i < LEN_2D; i++) {
        for (int j = 0; j <= i - 1; j++) {
            a[i] += bb[j][i] * a[i-j-1];
        }
    }

    FIXED: Launches kernels SEQUENTIALLY for each i to respect RAW dependencies.
    Each kernel parallelizes over j values.
    """
    a = a.contiguous()
    bb = bb.contiguous()

    n = a.size(0)

    if n <= 1:
        return a

    BLOCK_SIZE = 128

    # Process each i value SEQUENTIALLY
    for i in range(1, n):
        # Number of j values to process (j from 0 to i-1)
        num_j = i

        # Calculate grid size for this i
        grid_size = triton.cdiv(num_j, BLOCK_SIZE)

        # Launch kernel for this specific i value
        s118_kernel[(grid_size,)](
            a_ptr=a,
            bb_ptr=bb,
            n=n,
            i_val=i,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return a
