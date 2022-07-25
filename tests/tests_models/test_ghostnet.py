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


class TestGhostBottleneckModule(BaseLayer):
    def test_layer_constructor(
        self,
    ):
        layer1 = GhostBottleneckModule(
            dwkernel=3,
            strides=1,
            exp=16,
            out=16,
            ratio=2,
            use_se=False,
        )
        layer2 = GhostBottleneckModule(
            dwkernel=3,
            strides=2,
            exp=16,
            out=16,
            ratio=2,
            use_se=False,
        )
        layer3 = GhostBottleneckModule(
            dwkernel=3,
            strides=1,
            exp=16,
            out=16,
            ratio=2,
            use_se=True,
        )
        layer4 = GhostBottleneckModule(
            dwkernel=3,
            strides=2,
            exp=16,
            out=16,
            ratio=2,
            use_se=True,
        )
        super().test_layer_constructor(layer1)
        super().test_layer_constructor(layer2)
        super().test_layer_constructor(layer3)
        super().test_layer_constructor(layer4)

    def test_layer(self, fmap):
        layer1 = GhostBottleneckModule(
            dwkernel=3,
            strides=1,
            exp=16,
            out=16,
            ratio=2,
            use_se=False,
        )
        layer2 = GhostBottleneckModule(
            dwkernel=3,
            strides=2,
            exp=16,
            out=16,
            ratio=2,
            use_se=False,
        )
        layer3 = GhostBottleneckModule(
            dwkernel=3,
            strides=1,
            exp=16,
            out=16,
            ratio=2,
            use_se=True,
        )
        layer4 = GhostBottleneckModule(
            dwkernel=3,
            strides=2,
            exp=16,
            out=16,
            ratio=2,
            use_se=True,
        )

        out1 = layer1(fmap)
        out2 = layer2(fmap)
        out3 = layer3(fmap)
        out4 = layer4(fmap)

        assert out1.shape.as_list() == [1, 224, 224, 16]
        assert out2.shape.as_list() == [1, 112, 112, 16]
        assert out3.shape.as_list() == [1, 224, 224, 16]
        assert out4.shape.as_list() == [1, 112, 112, 16]

    def test_config(self):
        layer1 = GhostBottleneckModule(
            dwkernel=3,
            strides=1,
            exp=16,
            out=16,
            ratio=2,
            use_se=False,
        )
        layer2 = GhostBottleneckModule(
            dwkernel=3,
            strides=2,
            exp=16,
            out=16,
            ratio=2,
            use_se=False,
        )
        layer3 = GhostBottleneckModule(
            dwkernel=3,
            strides=1,
            exp=16,
            out=16,
            ratio=2,
            use_se=True,
        )
        layer4 = GhostBottleneckModule(
            dwkernel=3,
            strides=2,
            exp=16,
            out=16,
            ratio=2,
            use_se=True,
        )
        super().test_config(layer1)
        super().test_config(layer2)
        super().test_config(layer3)
        super().test_config(layer4)


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
