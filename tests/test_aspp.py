import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.layers.aspp import ASPP
from tests.utils import BaseLayer


@pytest.fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


class TestASPP(BaseLayer):
    def test_layer_constructor(self):
        layer = ASPP(filters=32)

        assert isinstance(layer, Layer)

    def test_layer(self, fmap):
        layer = ASPP(filters=32)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 32]

    def test_config(self):
        pass
