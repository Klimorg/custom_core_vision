import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.layers.joint_pyramid_upsampling import JointPyramidUpsampling
from tests.utils import BaseLayer


@pytest.fixture
def fmap8():
    return np.random.rand(1, 28, 28, 3)


@pytest.fixture
def fmap16():
    return np.random.rand(1, 14, 14, 3)


@pytest.fixture
def fmap32():
    return np.random.rand(1, 7, 7, 3)


class TestJPU(BaseLayer):
    def test_layer_constructor(self):
        layer = JointPyramidUpsampling()

        assert isinstance(layer, Layer)

    def test_layer(self, fmap8, fmap16, fmap32):
        layer = JointPyramidUpsampling()

        out = layer([fmap8, fmap16, fmap32])

        assert out.shape.as_list() == [1, 28, 28, 256]

    def test_config(self):
        pass
