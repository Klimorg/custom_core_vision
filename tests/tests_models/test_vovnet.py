import numpy as np
import pytest
from pytest import fixture
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.vovnet import OSAModule, VoVNet
from tests.utils import BaseLayer, BaseModel


@fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


class TestOSAModule(BaseLayer):
    def test_layer_constructor(
        self,
    ):
        layer = OSAModule(filters_conv3x3=64, filters_conv1x1=64)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = OSAModule(filters_conv3x3=64, filters_conv1x1=64)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 64]

    def test_config(
        self,
    ):
        layer = OSAModule(filters_conv3x3=64, filters_conv1x1=64)
        super().test_config(layer)


@pytest.mark.parametrize(
    "filters_conv3x3, filters_conv1x1, block_repetitions, name",
    [
        ([64, 80, 96, 112], [128, 256, 384, 512], [1, 1, 1, 1], "VoVNet27"),
        ([128, 256, 192, 224], [256, 512, 768, 1024], [1, 1, 2, 2], "VoVNet39"),
        ([128, 256, 192, 224], [256, 512, 768, 1024], [1, 1, 4, 3], "VoVNet57"),
    ],
)
class TestConvNeXt(BaseModel):
    def test_model_constructor(
        self,
        filters_conv3x3,
        filters_conv1x1,
        block_repetitions,
        name,
    ):
        model = VoVNet(
            img_shape=[224, 224, 3],
            filters_conv3x3=filters_conv3x3,
            filters_conv1x1=filters_conv1x1,
            block_repetitions=block_repetitions,
            name=name,
        )
        super().test_model_constructor(model)

    def test_classification_backbone(
        self,
        fmap,
        filters_conv3x3,
        filters_conv1x1,
        block_repetitions,
        name,
    ):
        model = VoVNet(
            img_shape=[224, 224, 3],
            filters_conv3x3=filters_conv3x3,
            filters_conv1x1=filters_conv1x1,
            block_repetitions=block_repetitions,
            name=name,
        )

        backbone = model.get_classification_backbone()

        out = backbone(fmap)

        assert out.shape.as_list() == [1, 7, 7, filters_conv1x1[3]]

    def test_segmentation_backbone(
        self,
        fmap,
        filters_conv3x3,
        filters_conv1x1,
        block_repetitions,
        name,
    ):
        model = VoVNet(
            img_shape=[224, 224, 3],
            filters_conv3x3=filters_conv3x3,
            filters_conv1x1=filters_conv1x1,
            block_repetitions=block_repetitions,
            name=name,
        )

        backbone = model.get_segmentation_backbone()
        super().test_segmentation_backbone(fmap, backbone)
