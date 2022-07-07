import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pytest import fixture
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.backbone.resnet18 import ResNet18, ResNetBlock


@fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


def test_constructor():

    layer1 = ResNetBlock(filters=32, downsample=False)
    layer2 = ResNetBlock(filters=32, downsample=True)
    model = ResNet18()

    assert isinstance(layer1, Layer)
    assert isinstance(layer2, Layer)
    assert isinstance(model, Model)


# @given(
#     arrays(
#         dtype=np.float32,
#         elements=st.floats(0, 1, width=32),
#         shape=[1, 224, 224, 3],
#     ),
# )
def test_compute(fmap):
    model = ResNet18()
    out = model(fmap)

    assert out.shape.as_list() == [1, 7, 7, 512]


# class BaseModelTesting:
#     pass
