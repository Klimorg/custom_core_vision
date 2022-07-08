import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.layers.common_layers import ConvBNReLU, ConvGNReLU
from tests.utils import BaseLayer


@pytest.fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


class TestConvGNReLU(BaseLayer):
    def test_layer_constructor(self):
        layer = ConvGNReLU(filters=32, kernel_size=3)

        assert isinstance(layer, Layer)

    def test_layer(self, fmap):
        layer = ConvGNReLU(filters=32, kernel_size=3)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 32]
        assert (out.numpy() >= 0).all()

    def test_config(self):
        pass


class TestConvBNReLU(BaseLayer):
    def test_layer_constructor(self):
        layer = ConvBNReLU(filters=32, kernel_size=3)

        assert isinstance(layer, Layer)

    def test_layer(self, fmap):
        layer = ConvBNReLU(filters=32, kernel_size=3)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 32]
        assert (out.numpy() >= 0).all()

    def test_config(self):
        pass
