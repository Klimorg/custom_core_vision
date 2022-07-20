from typing import Any, Dict

import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.heads.ksac import KSAC
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


class TestKSAC(BaseLayer):
    def test_layer_constructor(self):
        layer = KSAC(n_classes=10, filters=128, ksac_filters=128)

        super().test_layer_constructor(layer)

    def test_layer(self, fmap4, fmap8, fmap16, fmap32):
        layer = KSAC(n_classes=10, filters=128, ksac_filters=128)

        out = layer([fmap4, fmap8, fmap16, fmap32])

        assert out.shape.as_list() == [1, 224, 224, 10]

    def test_config(self):
        layer = KSAC(n_classes=10, filters=128, ksac_filters=128)
        super().test_config(layer)
