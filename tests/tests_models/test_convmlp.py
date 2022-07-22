import numpy as np
import pytest
from pytest import fixture
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.convmlp import (
    BasicStage,
    ConvDownsample,
    ConvMLP,
    ConvMLPStage,
    ConvStage,
    ConvTokenizer,
    Mlp,
)
from tests.utils import BaseLayer, BaseModel


@fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


@fixture
def fmap2():
    return np.random.rand(1, 56, 56, 64)


@fixture
def fmap3():
    return np.random.rand(1, 28, 28, 64)


@fixture
def fmap4():
    return np.random.rand(1, 224, 224, 128)


@fixture
def fmap5():
    return np.random.rand(1, 14, 14, 128)


class TestConvTokenizer(BaseLayer):
    def test_layer_constructor(self):

        layer = ConvTokenizer()
        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = ConvTokenizer()

        out = layer(fmap)

        assert out.shape.as_list() == [1, 56, 56, 64]

    def test_config(self):
        layer = ConvTokenizer()

        super().test_config(layer)


class TestConvStage(BaseLayer):
    def test_layer_constructor(self):

        layer = ConvStage()
        super().test_layer_constructor(layer)

    def test_layer(self, fmap2):
        layer = ConvStage()

        out = layer(fmap2)

        assert out.shape.as_list() == [1, 28, 28, 64]

    def test_config(self):
        layer = ConvStage()

        super().test_config(layer)


class TestConvDownsample(BaseLayer):
    def test_layer_constructor(self):
        layer = ConvDownsample(filters=128)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap3):
        layer = ConvDownsample(filters=128)

        out = layer(fmap3)

        assert out.shape.as_list() == [1, 14, 14, 128]

    def test_config(self):
        layer = ConvDownsample(filters=128)
        super().test_config(layer)


class TestMlp(BaseLayer):
    def test_layer_constructor(self):
        layer = Mlp(fc1_units=128, fc2_units=128)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap):
        layer = Mlp(fc1_units=128, fc2_units=128)

        out = layer(fmap)

        assert out.shape.as_list() == [1, 224, 224, 128]

    def test_config(self):

        layer = Mlp(fc1_units=128, fc2_units=128)

        super().test_config(layer)


class TestConvMLPStage(BaseLayer):
    def test_layer_constructor(self):
        layer = ConvMLPStage(expansion_units=256, units=128)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap4):
        layer = ConvMLPStage(expansion_units=256, units=128)
        out = layer(fmap4)

        assert out.shape.as_list() == [1, 224, 224, 128]

    def test_config(self):
        layer = ConvMLPStage(expansion_units=256, units=128)

        super().test_config(layer)


class TestBasicStage(BaseLayer):
    def test_layer_constructor(self):
        layer = BasicStage(num_blocks=2, units=128)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap5):
        layer = BasicStage(num_blocks=2, units=128)
        out = layer(fmap5)

        assert out.shape.as_list() == [1, 7, 7, 256]

    def test_config(self):
        layer = BasicStage(num_blocks=2, units=128)

        super().test_config(layer)


@pytest.mark.parametrize(
    "channels, n_conv_blocks, num_blocks, units, mlp_ratios, name",
    [
        (64, 2, [2, 2, 2], [128, 128, 256, 512], [2, 2, 2], "ConvMLP-XS"),
        (64, 2, [2, 4, 2], [128, 128, 256, 512], [2, 2, 2], "ConvMLP-S"),
        (64, 3, [3, 6, 3], [128, 128, 256, 512], [3, 3, 3], "ConvMLP-M"),
        (96, 3, [4, 8, 3], [192, 192, 384, 768], [3, 3, 3], "ConvMLP-L"),
    ],
)
class TestConvNeXt(BaseModel):
    def test_model_constructor(
        self,
        channels,
        n_conv_blocks,
        num_blocks,
        units,
        mlp_ratios,
        name,
    ):
        model = ConvMLP(
            img_shape=[224, 224, 3],
            channels=channels,
            n_conv_blocks=n_conv_blocks,
            num_blocks=num_blocks,
            units=units,
            mlp_ratios=mlp_ratios,
            name=name,
        )

        super().test_model_constructor(model)

    def test_classification_backbone(
        self,
        fmap,
        channels,
        n_conv_blocks,
        num_blocks,
        units,
        mlp_ratios,
        name,
    ):
        model = ConvMLP(
            img_shape=[224, 224, 3],
            channels=channels,
            n_conv_blocks=n_conv_blocks,
            num_blocks=num_blocks,
            units=units,
            mlp_ratios=mlp_ratios,
            name=name,
        )
        backbone = model.get_classification_backbone()
        out = backbone(fmap)

        assert isinstance(backbone, Model)
        assert out.shape.as_list() == [1, 7, 7, units[3]]

    def test_segmentation_backbone(
        self,
        fmap,
        channels,
        n_conv_blocks,
        num_blocks,
        units,
        mlp_ratios,
        name,
    ):
        model = ConvMLP(
            img_shape=[224, 224, 3],
            channels=channels,
            n_conv_blocks=n_conv_blocks,
            num_blocks=num_blocks,
            units=units,
            mlp_ratios=mlp_ratios,
            name=name,
        )
        backbone = model.get_segmentation_backbone()

        super().test_segmentation_backbone(fmap=fmap, backbone=backbone)
