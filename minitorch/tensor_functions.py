"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple.

    Args:
    ----
        x: Any value.

    Returns:
    -------
        tuple: The value wrapped in a tuple if it is not already a tuple.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    """Base class for all functions that support autodifferentiation."""

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Perform the backward pass for the function.

        Args:
        ----
            ctx: Context object containing saved values.
            grad_out: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, ...]: Gradients of the input tensors.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Perform the forward pass for the function.

        Args:
        ----
            ctx: Context object to save values for backward pass.
            *inps: Input tensors.

        Returns:
        -------
            Tensor: Output tensor.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history.

        Args:
        ----
            *vals: Input tensors.

        Returns:
        -------
            Tensor: Output tensor with history for backpropagation.

        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negation.

        Args:
        ----
            ctx: Context object.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tensor: Gradient of the input tensor.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Inverse function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for inverse.

        Args:
        ----
            ctx: Context object.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Inverse of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for inverse.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tensor: Gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Addition function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for addition.

        Args:
        ----
            ctx: Context object.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Sum of the input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for addition.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the input tensors.

        """
        return grad_output, grad_output


class All(Function):
    """All function."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for all.

        Args:
        ----
            ctx: Context object.
            a: Input tensor.
            dim: Dimension to reduce.

        Returns:
        -------
            Tensor: Reduced tensor.

        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    """Multiplication function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for multiplication.

        Args:
        ----
            ctx: Context object.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Product of the input tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the input tensors.

        """
        t1, t2 = ctx.saved_values
        return (
            grad_output.f.mul_zip(grad_output, t2),
            grad_output.f.mul_zip(grad_output, t1),
        )


class Sigmoid(Function):
    """Sigmoid function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for sigmoid.

        Args:
        ----
            ctx: Context object.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Sigmoid of the input tensor.

        """
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sigmoid.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tensor: Gradient of the input tensor.

        """
        temp = ctx.saved_values[0]
        return temp * (-temp + 1.0) * grad_output


class ReLU(Function):
    """ReLU function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for ReLU.

        Args:
        ----
            ctx: Context object.
            t1: Input tensor.

        Returns:
        -------
            Tensor: ReLU of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for ReLU.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tensor: Gradient of the input tensor.

        """
        a = ctx.saved_values[0]
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    """Logarithm function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for logarithm.

        Args:
        ----
            ctx: Context object.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Logarithm of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for logarithm.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tensor: Gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """Exponential function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for exponential.

        Args:
        ----
            ctx: Context object.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Exponential of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for exponential.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tensor: Gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        temp = t1.f.exp_map(t1)
        return grad_output.f.mul_zip(temp, grad_output)


class Sum(Function):
    """Sum function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for sum.

        Args:
        ----
            ctx: Context object.
            t1: Input tensor.
            dim: Dimension to reduce.

        Returns:
        -------
            Tensor: Sum of the input tensor along the specified dimension.

        """
        ctx.save_for_backward(t1.shape, dim)
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for sum.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, float]: Gradient of the input tensor and a float.

        """
        shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    """Less than function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for less than.

        Args:
        ----
            ctx: Context object.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Result of element-wise less than comparison.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less than.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the input tensors.

        """
        t1, t2 = ctx.saved_values
        return t1.zeros(t1.shape), t2.zeros(t2.shape)


class EQ(Function):
    """Equal function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for equal.

        Args:
        ----
            ctx: Context object.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Result of element-wise equality comparison.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equal.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the input tensors.

        """
        t1, t2 = ctx.saved_values
        return t1.zeros(t1.shape), t2.zeros(t2.shape)


class IsClose(Function):
    """Is close function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for is close.

        Args:
        ----
            ctx: Context object.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Result of element-wise is close comparison.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    """Permute function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Forward pass for permute.

        Args:
        ----
            ctx: Context object.
            t1: Input tensor.
            order: Order of permutation.

        Returns:
        -------
            Tensor: Permuted tensor.

        """
        ctx.save_for_backward(order)
        return t1._new(t1._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for permute.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, float]: Permuted gradient of the input tensor and a float.

        """
        order = ctx.saved_values[0]
        order2 = [
            temp[0]
            for temp in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]

        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    """View function."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward pass for view.

        Args:
        ----
            ctx: Context object.
            a: Input tensor.
            shape: New shape.

        Returns:
        -------
            Tensor: Reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for view.

        Args:
        ----
            ctx: Context object.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, float]: Reshaped gradient of the input tensor and a float.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the gradient of a function using the central difference method.

    Args:
    ----
        f : function to compute the gradient of
        *vals : tensors to pass to the function
        arg : index of the argument to compute the gradient with respect to (default is 0)
        epsilon : small value to use for the finite difference (default is 1e-6)
        ind : index to compute the gradient at

    Returns:
    -------
        float : computed gradient at the specified index

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
