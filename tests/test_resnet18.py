from tensorflow.keras.layers import Layer

from core_vision.models.backbone.resnet18 import ResNetBlock


def test_constructor():

    layer1 = ResNetBlock(filters=32, downsample=False)
    layer2 = ResNetBlock(filters=32, downsample=True)

    assert isinstance(layer1, Layer)
    assert isinstance(layer2, Layer)
