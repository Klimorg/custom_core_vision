import numpy as np
import pytest
from pytest import fixture
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.mit import (
    CustomAttention,
    FFNAttentionBlock,
    Mlp,
    OverlapPatchEmbed,
    SquareReshape,
)
from tests.utils import BaseLayer, BaseModel


@fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


@fixture
def fmap2():
    return np.random.rand(1, 3136, 64)


@pytest.mark.parametrize(
    "patch_size, strides, emb_dim",
    [
        (7, 4, 64),
        (3, 2, 128),
        (3, 2, 320),
        (3, 2, 512),
    ],
)
class TestOverlapPatchEmbed(BaseLayer):
    def test_layer_constructor(self, patch_size, strides, emb_dim):

        layer = OverlapPatchEmbed(
            patch_size=patch_size,
            strides=strides,
            emb_dim=emb_dim,
        )
        super().test_layer_constructor(layer)

    def test_layer(self, fmap, patch_size, strides, emb_dim):
        layer = OverlapPatchEmbed(
            patch_size=patch_size,
            strides=strides,
            emb_dim=emb_dim,
        )

        out = layer(fmap)

        assert out.shape.as_list() == [1, (224 // strides) * (224 // strides), emb_dim]

    def test_config(self, patch_size, strides, emb_dim):
        layer = OverlapPatchEmbed(
            patch_size=patch_size,
            strides=strides,
            emb_dim=emb_dim,
        )

        super().test_config(layer)


class TestMlp(BaseLayer):
    def test_layer_constructor(self):

        layer = Mlp(fc1_units=8 * 64, fc2_units=64)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap2):
        layer = Mlp(fc1_units=8 * 64, fc2_units=64)

        out = layer(fmap2)

        assert out.shape.as_list() == [1, 3136, 64]

    def test_config(self):
        layer = Mlp(fc1_units=8 * 64, fc2_units=64)

        super().test_config(layer)


# @pytest.mark.parametrize(
#     "channels, n_conv_blocks, num_blocks, units, mlp_ratios, name",
#     [
#         (64, 2, [2, 2, 2], [128, 128, 256, 512], [2, 2, 2], "ConvMLP-XS"),
#         (64, 2, [2, 4, 2], [128, 128, 256, 512], [2, 2, 2], "ConvMLP-S"),
#         (64, 3, [3, 6, 3], [128, 128, 256, 512], [3, 3, 3], "ConvMLP-M"),
#         (96, 3, [4, 8, 3], [192, 192, 384, 768], [3, 3, 3], "ConvMLP-L"),
#     ],
# )
# class TestConvNeXt(BaseModel):
