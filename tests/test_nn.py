import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # TODO: Implement for Task 4.4.
    # Max-reduce on first dimension -> 1x3x4
    out = minitorch.nn.max(t, 0)
    assert_close(out[0, 0, 0], max([t[i, 0, 0] for i in range(2)]))
    assert out.shape == (1, 3, 4)
    # Max-reduce on second dimension -> 2x1x4
    out = minitorch.nn.max(t, 1)
    assert_close(out[0, 0, 0], max([t[0, i, 0] for i in range(3)]))
    assert out.shape == (2, 1, 4)
    # Max-reduce on third dimension -> 2x3x1
    out = minitorch.nn.max(t, 2)
    assert_close(out[0, 0, 0], max([t[0, 0, i] for i in range(4)]))
    assert out.shape == (2, 3, 1)

    # Check gradient using grad_check (adds small noise to avoid non-differentiable points)
    minitorch.grad_check(
        lambda t: minitorch.nn.max(t, 0), t + (minitorch.rand(t.shape) * 1e-4)
    )

    # Check gradient manually
    t.requires_grad_(True)
    out = minitorch.nn.max(t, 2)  # max along the last dimension
    out.sum().backward()

    assert t.grad

    # Gradient should be 1 for the maximum value and 0 for the rest
    for i in range(2):
        for j in range(3):
            for k in range(4):
                if t[i, j, k] == out[i, j, 0]:  # if it's the maximum value
                    assert_close(t.grad[i, j, k], 1.0)
                else:
                    assert_close(t.grad[i, j, k], 0.0)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
