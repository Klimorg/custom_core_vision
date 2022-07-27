import numpy as np
import pytest
from pytest import fixture
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.mit import (
    EfficientSelfAttention,
    FFNAttentionBlock,
    MixFFN,
    MixTransformer,
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


@fixture
def fmap3():
    return np.random.rand(1, 3136, 32)


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


class TestMixFFN(BaseLayer):
    def test_layer_constructor(self):

        layer = MixFFN(fc1_units=8 * 64, fc2_units=64)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap2):
        layer = MixFFN(fc1_units=8 * 64, fc2_units=64)

        out = layer(fmap2)

        assert out.shape.as_list() == [1, 3136, 64]

    def test_config(self):
        layer = MixFFN(fc1_units=8 * 64, fc2_units=64)

        super().test_config(layer)


class TestEfficientSelfAttention(BaseLayer):
    def test_layer_constructor(self):

        layer = EfficientSelfAttention(units=64)
        super().test_layer_constructor(layer)

    def test_layer(self, fmap2):
        layer = EfficientSelfAttention(units=64)

        out = layer(fmap2)

        assert out.shape.as_list() == [1, 3136, 64]

    def test_config(self):
        layer = EfficientSelfAttention(units=64)

        super().test_config(layer)


class TestSquareReshape(BaseLayer):
    def test_layer_constructor(self):
        layer = SquareReshape()
        super().test_layer_constructor(layer)

    def test_layer(self, fmap2):
        layer = SquareReshape()
        out = layer(fmap2)

        assert out.shape.as_list() == [1, 56, 56, 64]

    def test_config(self):
        layer = SquareReshape()
        super().test_config(layer)


@pytest.mark.parametrize(
    "units, num_heads, attn_reduction_ratio",
    [
        (32, 1, 8),
    ],
)
class TestFFNAttentionBlock(BaseLayer):
    def test_layer_constructor(self, units, num_heads, attn_reduction_ratio):
        layer = FFNAttentionBlock(
            units=units,
            num_heads=num_heads,
            attn_reduction_ratio=attn_reduction_ratio,
        )

        super().test_layer_constructor(layer)

    def test_layer(self, fmap3, units, num_heads, attn_reduction_ratio):
        layer = FFNAttentionBlock(
            units=units,
            num_heads=num_heads,
            attn_reduction_ratio=attn_reduction_ratio,
        )

        out = layer(fmap3)
        assert out.shape.as_list() == [1, 3136, 32]

    def test_config(self, units, num_heads, attn_reduction_ratio):
        layer = FFNAttentionBlock(
            units=units,
            num_heads=num_heads,
            attn_reduction_ratio=attn_reduction_ratio,
        )
        super().test_config(layer)


@pytest.mark.parametrize(
    "patch_size, strides, emb_dims, num_heads, mlp_ratios, stochastic_depth_rate, attn_reduction_ratios, depths, name",
    [
        (
            [7, 3, 3, 3],
            [4, 2, 2, 2],
            [32, 64, 160, 256],
            [1, 2, 5, 8],
            [8, 8, 4, 4],
            0.1,
            [8, 4, 2, 1],
            [2, 2, 2, 2],
            "MiT-B0",
        ),
        (
            [7, 3, 3, 3],
            [4, 2, 2, 2],
            [64, 128, 320, 512],
            [1, 2, 5, 8],
            [8, 8, 4, 4],
            0.1,
            [8, 4, 2, 1],
            [2, 2, 2, 2],
            "MiT-B1",
        ),
        (
            [7, 3, 3, 3],
            [4, 2, 2, 2],
            [64, 128, 320, 512],
            [1, 2, 5, 8],
            [8, 8, 4, 4],
            0.1,
            [8, 4, 2, 1],
            [3, 3, 6, 3],
            "MiT-B2",
        ),
        (
            [7, 3, 3, 3],
            [4, 2, 2, 2],
            [64, 128, 320, 512],
            [1, 2, 5, 8],
            [8, 8, 4, 4],
            0.1,
            [8, 4, 2, 1],
            [3, 3, 18, 3],
            "MiT-B3",
        ),
        (
            [7, 3, 3, 3],
            [4, 2, 2, 2],
            [64, 128, 320, 512],
            [1, 2, 5, 8],
            [8, 8, 4, 4],
            0.1,
            [8, 4, 2, 1],
            [3, 8, 27, 3],
            "MiT-B4",
        ),
        (
            [7, 3, 3, 3],
            [4, 2, 2, 2],
            [64, 128, 320, 512],
            [1, 2, 5, 8],
            [8, 8, 4, 4],
            0.1,
            [8, 4, 2, 1],
            [3, 6, 40, 3],
            "MiT-B5",
        ),
    ],
)
class TestMixTransformer(BaseModel):
    def test_model_constructor(
        self,
        patch_size,
        strides,
        emb_dims,
        num_heads,
        mlp_ratios,
        stochastic_depth_rate,
        attn_reduction_ratios,
        depths,
        name,
    ):
        model = MixTransformer(
            img_shape=[224, 224, 3],
            patch_size=patch_size,
            strides=strides,
            emb_dims=emb_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            stochastic_depth_rate=stochastic_depth_rate,
            attn_reduction_ratios=attn_reduction_ratios,
            depths=depths,
            name=name,
        )
        super().test_model_constructor(model)

    def test_classification_backbone(
        self,
        fmap,
        patch_size,
        strides,
        emb_dims,
        num_heads,
        mlp_ratios,
        stochastic_depth_rate,
        attn_reduction_ratios,
        depths,
        name,
    ):
        model = MixTransformer(
            img_shape=[224, 224, 3],
            patch_size=patch_size,
            strides=strides,
            emb_dims=emb_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            stochastic_depth_rate=stochastic_depth_rate,
            attn_reduction_ratios=attn_reduction_ratios,
            depths=depths,
            name=name,
        )

        backbone = model.get_classification_backbone()

        out = backbone(fmap)

        assert out.shape.as_list() == [1, 7, 7, emb_dims[3]]

    def test_segmentation_backbone(
        self,
        fmap,
        patch_size,
        strides,
        emb_dims,
        num_heads,
        mlp_ratios,
        stochastic_depth_rate,
        attn_reduction_ratios,
        depths,
        name,
    ):
        model = MixTransformer(
            img_shape=[224, 224, 3],
            patch_size=patch_size,
            strides=strides,
            emb_dims=emb_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            stochastic_depth_rate=stochastic_depth_rate,
            attn_reduction_ratios=attn_reduction_ratios,
            depths=depths,
            name=name,
        )

        backbone = model.get_segmentation_backbone()
        super().test_segmentation_backbone(fmap, backbone)
