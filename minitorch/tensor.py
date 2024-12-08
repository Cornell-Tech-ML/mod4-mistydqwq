"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        """Initialize the Tensor object.

        Args:
        ----
            v : tensor data
            back : history of operations, optional
            name : name of the tensor, optional
            backend : tensor backend, optional

        """
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set whether the tensor requires gradient computation.

        Args:
        ----
            x : boolean indicating if gradient is required

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient computation.

        Returns
        -------
            bool : True if gradient is required

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Convert the tensor to a numpy array.

        Returns
        -------
            numpy array representation of the tensor

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Ensure the input is a tensor.

        Args:
        ----
            b : input value

        Returns:
        -------
            Tensor : tensor representation of the input

        """
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float.

        Returns
        -------
            float : value of the tensor

        """
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data.

        Returns
        -------
            Tensor : contiguous tensor

        """
        return Copy.apply(self)

    def __repr__(self) -> str:
        """Return a string representation of the tensor.

        Returns
        -------
            str : string representation of the tensor

        """
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        """Get the value at the specified index.

        Args:
        ----
            key : index to get the value from

        Returns:
        -------
            float : value at the specified index

        """
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        """Set the value at the specified index.

        Args:
        ----
            key : index to set the value at
            val : value to set

        """
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    def _type_(self, backend: TensorBackend) -> None:
        """Set the backend type for the tensor.

        Args:
        ----
            backend : tensor backend

        """
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        """Create a new tensor with the same backend.

        Args:
        ----
            tensor_data : tensor data

        Returns:
        -------
            Tensor : new tensor

        """
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data.

        Args:
        ----
            storage : tensor storage
            shape : tensor shape
            strides : tensor strides, optional
            backend : tensor backend, optional

        Returns:
        -------
            Tensor : new tensor

        """
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Expand the tensor to match the shape of another tensor.

        Args:
        ----
            other : tensor to match the shape with

        Returns:
        -------
            Tensor : expanded tensor

        """
        if self.shape == other.shape:
            return other

        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a tensor filled with zeros.

        Args:
        ----
            shape : shape of the tensor, optional

        Returns:
        -------
            Tensor : tensor filled with zeros

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple.

        Returns
        -------
            Tuple : tuple of storage, shape, and strides

        """
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach the tensor from the computation graph.

        Returns
        -------
            Tensor : detached tensor

        """
        return Tensor(self._tensor, backend=self.backend)

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the derivative accumulated on this variable.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Check if the tensor is a leaf variable.

        Returns
        -------
            bool : True if the tensor is a leaf variable

        """
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the tensor is a constant.

        Returns
        -------
            bool : True if the tensor is a constant

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of the tensor.

        Returns
        -------
            Iterable : parent variables

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients.

        Args:
        ----
            d_output : gradient of the output

        Returns:
        -------
            Iterable : gradients of the inputs

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Compute the gradients of the tensor.

        Args:
        ----
            grad_output : gradient of the output, optional

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Divide the tensor by another tensor or scalar.

        Args:
        ----
            b : tensor or scalar to divide by

        Returns:
        -------
            Tensor : result of the division

        """
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Divide a scalar by the tensor.

        Args:
        ----
            b : scalar to divide by

        Returns:
        -------
            Tensor : result of the division

        """
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Perform matrix multiplication with another tensor.

        Args:
        ----
            b : tensor to multiply with

        Returns:
        -------
            Tensor : result of the matrix multiplication

        """
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Get the shape of the tensor.

        Returns
        -------
            UserShape : shape of the tensor

        """
        shape = self._tensor.shape
        if isinstance(shape, int):
            return (shape,)
        return shape

    def __add__(self, b: TensorLike) -> Tensor:
        """Add the tensor to another tensor or scalar.

        Args:
        ----
            b : tensor or scalar to add

        Returns:
        -------
            Tensor : result of the addition

        """
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        """Subtract another tensor or scalar from the tensor.

        Args:
        ----
            b : tensor or scalar to subtract

        Returns:
        -------
            Tensor : result of the subtraction

        """
        return Add.apply(self, Neg.apply(self._ensure_tensor(b)))

    def __mul__(self, b: TensorLike) -> Tensor:
        """Multiply the tensor by another tensor or scalar.

        Args:
        ----
            b : tensor or scalar to multiply by

        Returns:
        -------
            Tensor : result of the multiplication

        """
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: TensorLike) -> Tensor:
        """Check if the tensor is less than another tensor or scalar.

        Args:
        ----
            b : tensor or scalar to compare with

        Returns:
        -------
            Tensor : result of the comparison

        """
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        """Check if the tensor is equal to another tensor or scalar.

        Args:
        ----
            b : tensor or scalar to compare with

        Returns:
        -------
            Tensor : result of the comparison

        """
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        """Check if the tensor is greater than another tensor or scalar.

        Args:
        ----
            b : tensor or scalar to compare with

        Returns:
        -------
            Tensor : result of the comparison

        """
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        """Negate the tensor.

        Returns
        -------
            Tensor : negated tensor

        """
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        """Add a scalar to the tensor.

        Args:
        ----
            b : scalar to add

        Returns:
        -------
            Tensor : result of the addition

        """
        return Add.apply(self._ensure_tensor(b), self)

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Multiply the tensor by a scalar.

        Args:
        ----
            b : scalar to multiply by

        Returns:
        -------
            Tensor : result of the multiplication

        """
        return Mul.apply(self._ensure_tensor(b), self)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Check if all elements of the tensor are true.

        Args:
        ----
            dim : dimension to reduce, optional

        Returns:
        -------
            Tensor : result of the check

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, b: TensorLike) -> Tensor:
        """Check if the tensor is close to another tensor or scalar.

        Args:
        ----
            b : tensor or scalar to compare with

        Returns:
        -------
            Tensor : result of the check

        """
        return IsClose.apply(self, self._ensure_tensor(b))

    def sigmoid(self) -> Tensor:
        """Apply the sigmoid function to the tensor.

        Returns
        -------
            Tensor : result of the sigmoid function

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Apply the ReLU function to the tensor.

        Returns
        -------
            Tensor : result of the ReLU function

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Apply the logarithm function to the tensor.

        Returns
        -------
            Tensor : result of the logarithm function

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Apply the exponential function to the tensor.

        Returns
        -------
            Tensor : result of the exponential function

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum over dimension `dim`.

        Args:
        ----
            dim : dimension to reduce, optional

        Returns:
        -------
            Tensor : result of the sum

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean over dimension `dim`.

        Args:
        ----
            dim : dimension to reduce, optional

        Returns:
        -------
            Tensor : result of the mean

        """
        if dim is None:
            return self.sum() / float(operators.prod(self.shape))
        else:
            return self.sum(dim) / float(self.shape[dim])

    def permute(self, *order: int) -> Tensor:
        """Permute tensor dimensions to *order.

        Args:
        ----
            *order : permutation of the dimensions

        Returns:
        -------
            Tensor : permuted tensor

        """
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: Any) -> Tensor:
        """Reshape the tensor to the specified shape.

        Args:
        ----
            *shape : new shape

        Returns:
        -------
            Tensor : reshaped tensor

        """
        if not shape or (len(shape) == 1 and shape[0] is None):
            return View.apply(self, tensor(list(self.shape)))

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        return View.apply(self, tensor(list(shape)))

    def zero_grad_(self) -> None:
        """Reset the gradients of the tensor."""
        self.grad = None

    @property
    def size(self) -> int:
        """Get the size of the tensor.

        Returns
        -------
            int : size of the tensor

        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Get the number of dimensions of the tensor.

        Returns
        -------
            int : number of dimensions

        """
        return len(self.shape)
