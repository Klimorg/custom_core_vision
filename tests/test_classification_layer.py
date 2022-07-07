import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pytest import fixture
from tensorflow.keras.layers import Layer

from core_vision.layers.classification_layer import ClassificationHead


@fixture
def fmap():
    return np.random.rand(1, 32, 32, 512)


def test_constructor():
    layer = ClassificationHead(units=64, num_classes=10)

    assert isinstance(layer, Layer)


# @given(
#     arrays(
#         dtype=np.float32,
#         elements=st.floats(0, 1, width=32),
#         shape=[1, 32, 32, 512],
#     ),
# )
def test_compute(fmap):
    layer = ClassificationHead(units=64, num_classes=10)
    out = layer(fmap)

    assert out.shape.as_list()[1] == 10
