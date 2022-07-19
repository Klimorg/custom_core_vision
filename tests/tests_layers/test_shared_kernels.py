import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.layers.shared_kernels import KSAConv2D, SharedDilatedConv
from tests.utils import BaseLayer


@pytest.fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


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


class TestSharedDilatedConv(BaseLayer):
    def test_layer_constructor(self):
        layer = SharedDilatedConv(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer="he_uniform",
            use_bias=False,
        )

        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = SharedDilatedConv(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer="he_uniform",
            use_bias=False,
        )

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 128]
        assert (out.numpy() >= 0).all()

    def test_config(self):
        layer = SharedDilatedConv(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer="he_uniform",
            use_bias=False,
        )

        super().test_config(layer)


class TestKSAConv2D(BaseLayer):
    def test_layer_constructor(self):
        layer = KSAConv2D(filters=128)

        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = KSAConv2D(filters=128)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 128]
        assert (out.numpy() >= 0).all()

    def test_config(self):
        layer = KSAConv2D(filters=128)
        super().test_config(layer)
