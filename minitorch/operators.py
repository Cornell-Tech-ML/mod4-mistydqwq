"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable
from typing import Sequence

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if x is close to y."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Log function."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def log_back(x: float, b: float) -> float:
    """Log backpropagation."""
    return b / x


def inv(x: float) -> float:
    """Inverse function."""
    return 1.0 / x


def inv_back(x: float, b: float) -> float:
    """Inverse backpropagation."""
    return -b / (x * x)


def relu_back(x: float, b: float) -> float:
    """ReLU backpropagation."""
    return b if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], l: list) -> list:
    """Applies a given function to each element of an iterable"""
    return [fn(x) for x in l]


def zipWith(fn: Callable[[float, float], float], l1: list, l2: list) -> list:
    """Combines elements from two iterables using a given function"""
    return [fn(x, y) for x, y in zip(l1, l2)]


def reduce(fn: Callable[[float, float], float], l: list, init: float) -> float:
    """Reduces an iterable to a single value using a given function"""
    res = init
    for x in l:
        res = fn(res, x)
    return res


def negList(l: list) -> list:
    """Negate all elements in a list using map"""
    return map(lambda x: -x, l)


def addLists(l1: list, l2: list) -> list:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(lambda x, y: x + y, l1, l2)


def sum(l: list) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, l, 0)


def prod(l: Sequence[int]) -> float:
    """Calculate the product of all elements in a sequence using reduce"""
    return reduce(mul, list(l), 1)
