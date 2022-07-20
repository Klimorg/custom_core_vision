from typing import Any, Dict

import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.heads.jpu import JPU, DecoderAddon
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


class TestDecoderAddon(BaseLayer):
    def test_layer_constructor(self):
        layer = DecoderAddon()
        super().test_layer_constructor(layer)

    def test_layer(self, fmap4, fmap8):
        layer = DecoderAddon()

        out = layer([fmap8, fmap4])

        assert out.shape.as_list() == [1, 56, 56, 128]

    def test_config(self):
        layer = DecoderAddon()
        super().test_config(layer)


class TestJPU(BaseLayer):
    def test_layer_constructor(self):
        layer = JPU(n_classes=10)

        super().test_layer_constructor(layer)

    def test_layer(self, fmap4, fmap8, fmap16, fmap32):
        layer1 = JPU(n_classes=10)
        layer2 = JPU(n_classes=10, decoder=False)

        out1 = layer1([fmap4, fmap8, fmap16, fmap32])
        out2 = layer2([fmap4, fmap8, fmap16, fmap32])

        assert out1.shape.as_list() == [1, 224, 224, 10]
        assert out2.shape.as_list() == [1, 224, 224, 10]

    def test_config(self):
        layer = JPU(n_classes=10)
        super().test_config(layer)
