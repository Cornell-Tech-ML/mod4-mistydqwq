# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any, List, Tuple

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions of your tensor_data functions.
# If you get an error, read the docs for NUMBA to see what is allowed in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT compile a function for use on CUDA devices.

    Args:
    ----
        fn (Fn): The function to be JIT compiled.
        **kwargs (Any): Additional keyword arguments for Numba JIT compilation.

    Returns:
    -------
        Fn: The JIT compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator to JIT compile a CUDA kernel function.

    Args:
    ----
        fn (Callable): The function to be JIT compiled.
        **kwargs (Any): Additional keyword arguments for Numba JIT compilation.

    Returns:
    -------
        FakeCUDAKernel: The JIT compiled CUDA kernel function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK: int = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Create a CUDA-enabled map function that applies `fn` element-wise.

        Args:
        ----
            fn (Callable[[float], float]): The function to apply.

        Returns:
        -------
            MapProto: A function that applies `fn` to a tensor using CUDA.

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the CUDA kernel.
            threadsperblock: int = THREADS_PER_BLOCK
            blockspergrid: int = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Create a CUDA-enabled zip function that applies `fn` element-wise to two tensors.

        Args:
        ----
            fn (Callable[[float, float], float]): The function to apply.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that applies `fn` to two tensors using CUDA.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape: Shape = shape_broadcast(a.shape, b.shape)
            out: Tensor = a.zeros(c_shape)
            threadsperblock: int = THREADS_PER_BLOCK
            blockspergrid: int = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Create a CUDA-enabled reduce function that reduces a tensor along a dimension.

        Args:
        ----
            fn (Callable[[float, float], float]): The reduction function.
            start (float, optional): The initial value for the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that reduces a tensor using CUDA.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape: List[int] = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a: Tensor = a.zeros(tuple(out_shape))

            threadsperblock: int = 1024
            blockspergrid: int = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication of two tensors using CUDA.

        Args:
        ----
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.

        Returns:
        -------
            Tensor: The result of matrix multiplication.

        """
        # Ensure tensors are at least 3-dimensional
        both_2d: bool = False
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d = True
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d = both_2d and True

        ls: List[int] = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out: Tensor = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra cols
        blockspergrid: Tuple[int, int, int] = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock: Tuple[int, int, int] = (
            THREADS_PER_BLOCK,
            THREADS_PER_BLOCK,
            1,
        )

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Reshape back to 2D if both inputs were 2D
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function.

    Example:
    -------
        fn_map = tensor_map(fn)
        fn_map(out, ... )

    Args:
    ----
        fn (Callable[[float], float]): Function mapping floats to floats to apply.

    Returns:
    -------
        Callable: Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index: numba.cuda.devicearray.LocalArray = cuda.local.array(
            MAX_DIMS, numba.int32
        )
        in_index: numba.cuda.devicearray.LocalArray = cuda.local.array(
            MAX_DIMS, numba.int32
        )
        i: int = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # Implemented for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos: int = index_to_position(in_index, in_strides)
            out[i] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    """CUDA higher-order tensor zipWith (or map2) function.

    Example:
    -------
        fn_zip = tensor_zip(fn)
        fn_zip(out, ...)

    Args:
    ----
        fn (Callable[[float, float], float]): Function mapping two floats to a float.

    Returns:
    -------
        Callable: Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index: numba.cuda.devicearray.LocalArray = cuda.local.array(
            MAX_DIMS, numba.int32
        )
        a_index: numba.cuda.devicearray.LocalArray = cuda.local.array(
            MAX_DIMS, numba.int32
        )
        b_index: numba.cuda.devicearray.LocalArray = cuda.local.array(
            MAX_DIMS, numba.int32
        )
        i: int = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Implemented for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_pos: int = index_to_position(a_index, a_strides)
            b_pos: int = index_to_position(b_index, b_strides)
            out[i] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice sum kernel to prepare for reduce.

    Given an array of length `n` and `out` of size `n // blockDim`, it sums up each `blockDim` values into an output cell.

    Example:
    -------
        [a_1, a_2, ..., a_100] -> [sum(a_1 to a_31), sum(a_32 to a_63), ...]

    Note:
    ----
        Each block must perform the sum using shared memory.

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the input tensor.
        size (int): Length of the input tensor `a`.

    """
    BLOCK_DIM: int = 32

    cache: numba.cuda.devicearray.SharedArray = cuda.shared.array(
        BLOCK_DIM, numba.float64
    )
    i: int = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos: int = cuda.threadIdx.x
    # Implemented for Task 3.3.

    if i < size:
        cache[pos] = float(a[i])
    else:
        cache[pos] = 0.0

    cuda.syncthreads()

    temp = 1
    while temp < BLOCK_DIM:
        if pos % (2 * temp) == 0:
            cache[pos] += cache[pos + temp]
        temp *= 2
        cuda.syncthreads()

    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice function to test the sum kernel.

    Args:
    ----
        a (Tensor): Input tensor to be summed.

    Returns:
    -------
        TensorData: Output tensor containing partial sums.

    """
    (size,) = a.shape
    threadsperblock: int = THREADS_PER_BLOCK
    blockspergrid: int = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    out: TensorData = TensorData([0.0 for _ in range(blockspergrid)], (blockspergrid,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None
]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn (Callable[[float, float], float]): Reduction function mapping two floats to a float.

    Returns:
    -------
        Callable: Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM: int = 1024
        cache: numba.cuda.devicearray.SharedArray = cuda.shared.array(
            BLOCK_DIM, numba.float64
        )
        out_index: numba.cuda.devicearray.LocalArray = cuda.local.array(
            MAX_DIMS, numba.int32
        )
        out_pos: int = cuda.blockIdx.x
        pos: int = cuda.threadIdx.x

        # Implemented for Task 3.3.
        cache[pos] = reduce_value
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            op: int = index_to_position(out_index, out_strides)
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                a_pos: int = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[a_pos]
                cuda.syncthreads()

                temp: int = 0
                while 2**temp < BLOCK_DIM:
                    stride: int = 2**temp
                    if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
                        cache[pos] = fn(cache[pos], cache[pos + stride])
                    cuda.syncthreads()
                    temp += 1

                if pos == 0:
                    out[op] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square matrix multiplication kernel.

    Given storages `out`, `a`, and `b`, where both `a` and `b` have shape `[size, size]` with strides `[size, 1]`.

    Size is always less than or equal to 32.

    Requirements:

    - All data must first be moved to shared memory.
    - Only read each cell in `a` and `b` once.
    - Only write to global memory once per kernel.

    Compute:

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    out[i, j] += a[i, k] * b[k, j]

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the first input tensor.
        b (Storage): Storage for the second input tensor.
        size (int): The size of the square matrices.

    """
    BLOCK_DIM: int = 32
    # Implemented for Task 3.3.
    a_shared: numba.cuda.devicearray.SharedArray = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float64
    )
    b_shared: numba.cuda.devicearray.SharedArray = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float64
    )
    i: int = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j: int = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < size and j < size:
        a_shared[cuda.threadIdx.x, cuda.threadIdx.y] = a[i * size + cuda.threadIdx.y]
        b_shared[cuda.threadIdx.x, cuda.threadIdx.y] = b[cuda.threadIdx.x * size + j]
        cuda.syncthreads()

        temp: float = 0.0
        for k in range(size):
            temp += a_shared[cuda.threadIdx.x, k] * b_shared[k, cuda.threadIdx.y]

        out[i * size + j] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice function to test the matrix multiplication kernel.

    Args:
    ----
        a (Tensor): The first input tensor.
        b (Tensor): The second input tensor.

    Returns:
    -------
        TensorData: The output tensor after matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock: Tuple[int, int] = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid: Tuple[int, int] = (1, 1)
    out: TensorData = TensorData([0.0 for _ in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    - All data must be first moved to shared memory.
    - Only read each cell in `a` and `b` once.
    - Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as:

        assert a_shape[-1] == b_shape[-2]

    Args:
    ----
        out (Storage): Storage for the output tensor.
        out_shape (Shape): Shape of the output tensor.
        out_strides (Strides): Strides of the output tensor.
        out_size (int): Total size of the output tensor.
        a_storage (Storage): Storage for the first input tensor.
        a_shape (Shape): Shape of the first input tensor.
        a_strides (Strides): Strides of the first input tensor.
        b_storage (Storage): Storage for the second input tensor.
        b_shape (Shape): Shape of the second input tensor.
        b_strides (Strides): Strides of the second input tensor.

    """
    a_batch_stride: int = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride: int = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch: int = cuda.blockIdx.z

    BLOCK_DIM: int = 32
    a_shared: numba.cuda.devicearray.SharedArray = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float64
    )
    b_shared: numba.cuda.devicearray.SharedArray = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float64
    )

    # The final position c[i, j] (output matrix position)
    i: int = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j: int = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    tx: int = cuda.threadIdx.x
    ty: int = cuda.threadIdx.y

    # Accumulator for out[i,j] -> this thread will compute this value.
    res: float = 0.0

    k_size: int = a_shape[-1]
    a_rows: int = a_shape[-2]
    b_cols: int = b_shape[-1]

    for k_block in range(0, k_size, BLOCK_DIM):
        tempi: int = i
        tempj: int = k_block + ty
        if tempi < a_rows and tempj < k_size:
            a_shared[tx, ty] = a_storage[
                batch * a_batch_stride + tempi * a_strides[-2] + tempj * a_strides[-1]
            ]
        else:
            a_shared[tx, ty] = 0.0

        tempbi: int = k_block + tx
        tempbj: int = j
        if tempbi < k_size and tempbj < b_cols:
            b_shared[tx, ty] = b_storage[
                batch * b_batch_stride + tempbi * b_strides[-2] + tempbj * b_strides[-1]
            ]
        else:
            b_shared[tx, ty] = 0.0
        cuda.syncthreads()
        for k in range(min(BLOCK_DIM, k_size - k_block)):
            res += a_shared[tx, k] * b_shared[k, ty]
        cuda.syncthreads()

    if i < a_rows and j < b_cols:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = res


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
