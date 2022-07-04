from hypothesis import given
from hypothesis import strategies as st
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.backbone.resnet18 import ResNet18, ResNetBlock


def test_constructor():

    layer1 = ResNetBlock(filters=32, downsample=False)
    layer2 = ResNetBlock(filters=32, downsample=True)
    model = ResNet18()

    assert isinstance(layer1, Layer)
    assert isinstance(layer2, Layer)
    assert isinstance(model, Model)
