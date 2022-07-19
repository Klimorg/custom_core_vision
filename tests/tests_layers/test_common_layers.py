import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.layers.common_layers import (
    BNConvReLU,
    ConvBNReLU,
    ConvGNReLU,
    InvertedResidualBottleneck2D,
    SepConvBNReLU,
)
from tests.utils import BaseLayer


@pytest.fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


class TestConvGNReLU(BaseLayer):
    def test_layer_constructor(self):
        layer = ConvGNReLU(filters=32, kernel_size=3)

        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = ConvGNReLU(filters=32, kernel_size=3)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 32]
        assert (out.numpy() >= 0).all()

    def test_config(self):
        layer = ConvGNReLU(filters=32, kernel_size=3)
        super().test_config(layer)


class TestConvBNReLU(BaseLayer):
    def test_layer_constructor(self):
        layer = ConvBNReLU(filters=32, kernel_size=3)

        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = ConvBNReLU(filters=32, kernel_size=3)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 32]
        assert (out.numpy() >= 0).all()

    def test_config(self):
        layer = ConvBNReLU(filters=32, kernel_size=3)
        super().test_config(layer)


class TestBNConvReLU(BaseLayer):
    def test_layer_constructor(self):
        layer = BNConvReLU(filters=32, kernel_size=3)

        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = BNConvReLU(filters=32, kernel_size=3)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 32]
        assert (out.numpy() >= 0).all()

    def test_config(self):
        layer = BNConvReLU(filters=32, kernel_size=3)
        super().test_config(layer)


class TestSepConvBNReLU(BaseLayer):
    def test_layer_constructor(self):
        layer = SepConvBNReLU(filters=32, kernel_size=3)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = SepConvBNReLU(filters=32, kernel_size=3)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 32]
        assert (out.numpy() >= 0).all()

    def test_config(self):
        layer = SepConvBNReLU(filters=32, kernel_size=3)
        super().test_config(layer)


class TestInvertedResidualBottleneck2D(BaseLayer):
    def test_layer_constructor(self):
        layer1 = InvertedResidualBottleneck2D(
            expansion_rate=2,
            filters=3,
            strides=1,
            skip_connection=True,
        )
        layer2 = InvertedResidualBottleneck2D(
            expansion_rate=2,
            filters=3,
            strides=2,
            skip_connection=False,
        )

        super().test_layer_constructor(layer1)
        super().test_layer_constructor(layer2)

    def test_layer(self, fmap):
        layer1 = InvertedResidualBottleneck2D(
            expansion_rate=2,
            filters=3,
            strides=1,
            skip_connection=True,
        )
        layer2 = InvertedResidualBottleneck2D(
            expansion_rate=2,
            filters=32,
            strides=2,
            skip_connection=False,
        )

        out1 = layer1(fmap)
        out2 = layer2(fmap)

        assert out1.shape.as_list() == [1, 224, 224, 3]
        assert out2.shape.as_list() == [1, 112, 112, 32]

    def test_config(self):
        layer1 = InvertedResidualBottleneck2D(
            expansion_rate=2,
            filters=3,
            strides=1,
            skip_connection=True,
        )
        layer2 = InvertedResidualBottleneck2D(
            expansion_rate=2,
            filters=32,
            strides=2,
            skip_connection=False,
        )

        super().test_config(layer1)
        super().test_config(layer2)
