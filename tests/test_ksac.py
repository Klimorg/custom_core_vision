import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.layers.shared_kernels import KSAConv2D, SharedDilatedConv
from tests.utils import BaseLayer


@pytest.fixture
def fmap4():
    return np.random.rand(1, 56, 56, 3)


@pytest.fixture
def fmap8():
    return np.random.rand(1, 28, 28, 3)


@pytest.fixture
def fmap16():
    return np.random.rand(1, 14, 14, 3)


@pytest.fixture
def fmap32():
    return np.random.rand(1, 7, 7, 3)
