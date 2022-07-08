import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pytest import fixture
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.convnext import ConvNext, ConvNextBlock, ConvNeXtLayer


@fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


def test_layer_constructor():

    layer1 = ConvNextBlock(filters=32)
    layer2 = ConvNeXtLayer(filters=32, num_blocks=4)

    assert isinstance(layer1, Layer)
    assert isinstance(layer2, Layer)


@pytest.mark.parametrize(
    "filters, num_blocks, name",
    [
        ([128, 256, 384, 768], [3, 3, 27, 3], "convnext-b"),
        ([192, 384, 768, 1536], [3, 3, 27, 3], "convnext-l"),
        ([96, 182, 384, 768], [3, 3, 27, 3], "convnext-s"),
        ([96, 182, 384, 768], [3, 3, 9, 3], "convnext-t"),
        ([256, 512, 1024, 2048], [3, 3, 27, 3], "convnext-xl"),
    ],
)
def test_model_constructor(filters, num_blocks, name):
    model = ConvNext(filters=filters, num_blocks=num_blocks, name=name)

    assert isinstance(model, Model)


@pytest.mark.parametrize(
    "filters, num_blocks, name",
    [
        ([128, 256, 384, 768], [3, 3, 27, 3], "convnext-b"),
        ([192, 384, 768, 1536], [3, 3, 27, 3], "convnext-l"),
        ([96, 182, 384, 768], [3, 3, 27, 3], "convnext-s"),
        ([96, 182, 384, 768], [3, 3, 9, 3], "convnext-t"),
        ([256, 512, 1024, 2048], [3, 3, 27, 3], "convnext-xl"),
    ],
)
def test_compute(fmap, filters, num_blocks, name):
    model = ConvNext(filters=filters, num_blocks=num_blocks, name=name)
    out = model(fmap)

    assert out.shape.as_list() == [1, 7, 7, filters[3]]
