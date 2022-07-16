from typing import Any, Dict

import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.layers.feature_pyramids import FeaturePyramidNetwork, SemanticHeadFPN
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


class TestFPN(BaseLayer):
    def test_layer_constructor(self):
        layer = FeaturePyramidNetwork()

        assert isinstance(layer, Layer)

    def test_layer(self, fmap4, fmap8, fmap16, fmap32):
        layer = FeaturePyramidNetwork()

        out = layer([fmap4, fmap8, fmap16, fmap32])

        assert out[0].shape.as_list() == [1, 56, 56, 256]
        assert out[1].shape.as_list() == [1, 28, 28, 256]
        assert out[2].shape.as_list() == [1, 14, 14, 256]
        assert out[3].shape.as_list() == [1, 7, 7, 256]

    def test_config(self):
        pass


class TestSemanticFPN(BaseLayer):
    def test_layer_constructor(self):
        layer = SemanticHeadFPN()

        assert isinstance(layer, Layer)

    def test_layer(self, fmap4, fmap8, fmap16, fmap32):
        layer = SemanticHeadFPN()

        out = layer([fmap4, fmap8, fmap16, fmap32])

        assert out.shape.as_list() == [1, 56, 56, 512]

    def test_config(self):
        layer = SemanticHeadFPN()

        config = layer.get_config()
        assert isinstance(config, Dict)
