import numpy as np
import pytest
from pytest import fixture
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.ghostnet import (
    GhostBottleneckModule,
    GhostModule,
    GhostNet,
    SqueezeAndExcite,
)
from tests.utils import BaseLayer, BaseModel


@fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


@fixture
def fmap2():
    return np.random.rand(1, 224, 224, 64)


class TestSqueezeAndExcite(BaseLayer):
    def test_layer_constructor(
        self,
    ):
        layer = SqueezeAndExcite(ratio=2, filters=64)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap2):
        layer = SqueezeAndExcite(ratio=2, filters=64)

        out = layer(fmap2)

        assert out.shape.as_list() == [1, 224, 224, 64]

    def test_config(self):
        layer = SqueezeAndExcite(ratio=2, filters=64)
        super().test_config(layer)


class TestGhostModule(BaseLayer):
    def test_layer_constructor(
        self,
    ):
        layer = GhostModule(out=16, ratio=2, convkernel=3, dwkernel=3)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = GhostModule(out=16, ratio=2, convkernel=3, dwkernel=3)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 16]

    def test_config(self):
        layer = GhostModule(out=16, ratio=2, convkernel=3, dwkernel=3)
        super().test_config(layer)


@pytest.mark.parametrize(
    "strides, use_se",
    [
        (1, False),
        (1, True),
        (2, False),
        (2, True),
    ],
)
class TestGhostBottleneckModule(BaseLayer):
    def test_layer_constructor(self, strides, use_se):
        layer = GhostBottleneckModule(
            dwkernel=3,
            strides=strides,
            exp=16,
            out=16,
            ratio=2,
            use_se=use_se,
        )

        super().test_layer_constructor(layer)

    def test_layer(self, fmap, strides, use_se):
        layer = GhostBottleneckModule(
            dwkernel=3,
            strides=strides,
            exp=16,
            out=16,
            ratio=2,
            use_se=use_se,
        )

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224 // strides, 224 // strides, 16]

    def test_config(self, strides, use_se):
        layer = GhostBottleneckModule(
            dwkernel=3,
            strides=strides,
            exp=16,
            out=16,
            ratio=2,
            use_se=use_se,
        )

        super().test_config(layer)


class TestGhostNet(BaseModel):
    def test_model_constructor(self):
        model = GhostNet(img_shape=[224, 224, 3])
        super().test_model_constructor(model)

    def test_classification_backbone(self, fmap):
        model = GhostNet(img_shape=[224, 224, 3])

        backbone = model.get_classification_backbone()

        out = backbone(fmap)

        assert out.shape.as_list() == [1, 7, 7, 960]

    def test_segmentation_backbone(self, fmap):
        model = GhostNet(img_shape=[224, 224, 3])
        backbone = model.get_segmentation_backbone()

        super().test_segmentation_backbone(fmap, backbone)
