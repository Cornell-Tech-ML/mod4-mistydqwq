from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    Inv,
    Mul,
    ScalarFunction,
    Add,
    Log,
    Exp,
    Sigmoid,
    ReLU,
    Lt,
    Neg,
    Eq,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn (Optional[Type[ScalarFunction]]): The last Function that was called.
        ctx (Optional[Context]): The context for that Function.
        inputs (Sequence[Scalar]): The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.

    Attributes
    ----------
        data (float): The scalar value.
        history (Optional[ScalarHistory]): The history of operations that created this scalar.
        derivative (Optional[float]): The derivative of this scalar.
        name (str): The name of this scalar.
        unique_id (int): The unique identifier for this scalar.

    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        """Returns a string representation of the scalar.

        Returns
        -------
            str: The string representation of the scalar.

        """
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiplies this scalar by another scalar or scalar-like value.

        Args:
        ----
            b (ScalarLike): The value to multiply by.

        Returns:
        -------
            Scalar: The result of the multiplication.

        """
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Divides this scalar by another scalar or scalar-like value.

        Args:
        ----
            b (ScalarLike): The value to divide by.

        Returns:
        -------
            Scalar: The result of the division.

        """
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Divides another scalar or scalar-like value by this scalar.

        Args:
        ----
            b (ScalarLike): The value to be divided.

        Returns:
        -------
            Scalar: The result of the division.

        """
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        """Returns the boolean value of this scalar.

        Returns
        -------
            bool: The boolean value of this scalar.

        """
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Adds another scalar or scalar-like value to this scalar.

        Args:
        ----
            b (ScalarLike): The value to add.

        Returns:
        -------
            Scalar: The result of the addition.

        """
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Multiplies another scalar or scalar-like value by this scalar.

        Args:
        ----
            b (ScalarLike): The value to multiply.

        Returns:
        -------
            Scalar: The result of the multiplication.

        """
        return self * b

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x (Any): The value to be accumulated.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`).

        Returns
        -------
            bool: True if this variable is a leaf.

        """
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant.

        Returns
        -------
            bool: True if this variable is a constant.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables of this variable.

        Returns
        -------
            Iterable[Variable]: The parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for this variable.

        Args:
        ----
            d_output (Any): The derivative of the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing the parent variable and its corresponding derivative.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None
        gradients = h.last_fn.backward(h.ctx, d_output)

        def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
            if isinstance(x, tuple):
                return x
            return (x,)

        return zip(h.inputs, wrap_tuple(gradients))

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (Optional[float]): Starting derivative to backpropagate through the model
                                        (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Less than comparison between this scalar and another scalar or scalar-like value.

        Args:
        ----
            b (ScalarLike): The value to compare.

        Returns:
        -------
            Scalar: The result of the comparison.

        """
        return Lt.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Greater than comparison between this scalar and another scalar or scalar-like value.

        Args:
        ----
            b (ScalarLike): The value to compare.

        Returns:
        -------
            Scalar: The result of the comparison.

        """
        return Lt.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Equality comparison between this scalar and another scalar or scalar-like value.

        Args:
        ----
            b (ScalarLike): The value to compare.

        Returns:
        -------
            Scalar: The result of the comparison.

        """
        return Eq.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtracts another scalar or scalar-like value from this scalar.

        Args:
        ----
            b (ScalarLike): The value to subtract.

        Returns:
        -------
            Scalar: The result of the subtraction.

        """
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Negates this scalar.

        Returns
        -------
            Scalar: The negated scalar.

        """
        return Neg.apply(self)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Adds another scalar or scalar-like value to this scalar.

        Args:
        ----
            b (ScalarLike): The value to add.

        Returns:
        -------
            Scalar: The result of the addition.

        """
        return Add.apply(self, b)

    def log(self) -> Scalar:
        """Computes the natural logarithm of this scalar.

        Returns
        -------
            Scalar: The natural logarithm of this scalar.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Computes the exponential of this scalar.

        Returns
        -------
            Scalar: The exponential of this scalar.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Computes the sigmoid of this scalar.

        Returns
        -------
            Scalar: The sigmoid of this scalar.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Computes the ReLU of this scalar.

        Returns
        -------
            Scalar: The ReLU of this scalar.

        """
        return ReLU.apply(self)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f (Any): Function from n-scalars to 1-scalar.
        *scalars (Scalar): n input scalar values.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
