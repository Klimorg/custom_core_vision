from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from einops.layers.tensorflow import Rearrange
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Input,
    Layer,
    LayerNormalization,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow_addons.layers import StochasticDepth

from core_vision.models.utils import TFModel

# Referred from: github.com:rwightman/pytorch-image-models.
# https://keras.io/examples/vision/cct/#stochastic-depth-for-regularization
# @tf.keras.utils.register_keras_serializable()
# class StochasticDepth(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         drop_prop,
#         *args,
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)

#         self.drop_prob = drop_prop

#     def call(self, inputs, training=None) -> tf.Tensor:
#         if training:
#             keep_prob = tf.cast(1 - self.drop_prob, dtype=inputs.dtype)
#             shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
#             random_tensor = keep_prob + tf.random.uniform(
#                 shape,
#                 0,
#                 1,
#                 dtype=inputs.dtype,
#             )
#             random_tensor = tf.floor(random_tensor)
#             return (inputs / keep_prob) * random_tensor
#         return inputs

#     def get_config(self) -> Dict[str, Any]:

#         config = super().get_config()
#         config.update({"drop_prob": self.drop_prob})
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


# # @tf.keras.utils.register_keras_serializable()
# class Identity(tf.keras.layers.Layer):
#     def __init__(self) -> None:
#         super().__init__(name="IdentityTF")

#     def call(self, inputs) -> tf.Tensor:
#         return inputs

#     def get_config(self) -> Dict[str, Any]:
#         config = super().get_config()
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


@tf.keras.utils.register_keras_serializable()
class OverlapPatchEmbed(Layer):
    def __init__(
        self,
        patch_size: int,
        strides: int,
        emb_dim: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            patch_size (int): _description_
            strides (int): _description_
            emb_dim (int): _description_
            l2_regul (float, optional): _description_. Defaults to 1e-4.

        Info: Input[B, H, W, C] --> Output[B, T, C].
            With T:=(H // strides) * (H // strides)
        """

        super().__init__(*args, **kwargs)

        self.patch_size = patch_size
        self.strides = strides
        self.emb_dim = emb_dim
        self.l2_regul = l2_regul

        self.norm = LayerNormalization()

    def build(self, input_shape) -> None:

        _, height, width, *_ = input_shape

        self.H = height // self.strides
        self.W = width // self.strides

        self.proj = Conv2D(
            filters=self.emb_dim,
            kernel_size=self.patch_size,
            strides=self.strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.reshape = Rearrange("b h w c -> b (h w) c")

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = self.proj(inputs)
        fmap = self.reshape(fmap)
        return self.norm(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "strides": self.strides,
                "emb_dim": self.emb_dim,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MixFFN(Layer):
    def __init__(
        self,
        fc1_units: int,
        fc2_units: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            fc1_units (int): _description_
            fc2_units (int): _description_
            l2_regul (float, optional): _description_. Defaults to 1e-4.
        """

        super().__init__(*args, **kwargs)

        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.l2_regul = l2_regul

        self.gelu = tf.keras.activations.gelu

    def build(self, input_shape) -> None:

        _, tensors, _ = input_shape

        height = int(tf.sqrt(float(tensors)))
        width = int(tf.sqrt(float(tensors)))

        self.square_reshape = Rearrange("b (h w) c -> b h w c", h=height, w=width)
        self.wide_reshape = Rearrange("b h w c -> b (h w) c")

        self.fc1 = Dense(
            self.fc1_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.fc2 = Dense(
            self.fc2_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.depth_conv = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = self.fc1(inputs)

        fmap = self.square_reshape(fmap)
        fmap = self.depth_conv(fmap)
        fmap = self.wide_reshape(fmap)

        fmap = self.gelu(fmap)
        return self.fc2(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "fc1_units": self.fc1_units,
                "fc2_units": self.fc2_units,
                "l2_regularization": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class EfficientSelfAttention(Layer):
    def __init__(
        self,
        units: int,
        num_heads: int = 1,
        attn_reduction_ratio: int = 8,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        assert (
            units % num_heads == 0
        ), f"dim {units} should be divided by num_heads {num_heads}."

        self.units = units
        self.num_heads = num_heads
        self.attn_reduction_ratio = attn_reduction_ratio
        self.l2_regul = l2_regul

        self.head_dims = units // num_heads
        self.scale = 1 / tf.sqrt(float(self.head_dims))

        self.softmax = tf.keras.activations.softmax

    def build(self, input_shape) -> None:

        _, tensors, _ = input_shape

        height = int(tf.sqrt(float(tensors)))
        width = int(tf.sqrt(float(tensors)))

        self.heads_reshape = Rearrange(
            "batch units (head_dims num_heads) -> batch units num_heads head_dims",
            head_dims=self.head_dims,
            num_heads=self.num_heads,
        )
        self.permute = Rearrange(
            "batch units num_heads head_dims -> batch num_heads units head_dims",
        )

        self.square_reshape = Rearrange("b (h w) c -> b h w c", h=height, w=width)

        self.height_width_merge = Rearrange("b h w c -> b (h w) c")
        self.heads_channels_merge = Rearrange("b n h c -> b n (h c)")

        self.kv_reshape = Rearrange(
            "batch units (f num_heads head_dims) -> batch units f num_heads head_dims",
            f=2,
            num_heads=self.num_heads,
            head_dims=self.head_dims,
        )

        self.query = Dense(
            self.units,
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.key_value = Dense(
            self.units * 2,
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.proj = Dense(
            self.units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        if self.attn_reduction_ratio > 1:
            self.attn_conv = Conv2D(
                filters=self.units,
                kernel_size=self.attn_reduction_ratio,
                strides=self.attn_reduction_ratio,
                padding="same",
                use_bias=False,
                kernel_initializer="he_uniform",
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            )
            self.norm = LayerNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = inputs

        queries = self.query(inputs)

        queries = self.heads_reshape(queries)
        queries = self.permute(queries)

        if self.attn_reduction_ratio > 1:
            fmap = self.square_reshape(fmap)
            fmap = self.attn_conv(fmap)
            fmap = self.height_width_merge(fmap)
            fmap = self.norm(fmap)

        fmap = self.key_value(fmap)
        fmap = self.kv_reshape(fmap)
        fmap = tf.transpose(fmap, perm=[2, 0, 3, 1, 4])
        keys, values = tf.split(fmap, num_or_size_splits=2)
        keys = tf.squeeze(keys, axis=0)
        values = tf.squeeze(values, axis=0)

        attn = tf.matmul(queries, keys, transpose_b=True) * self.scale
        attn = self.softmax(attn)

        x = tf.matmul(attn, values)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = self.heads_channels_merge(x)

        return self.proj(x)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_heads": self.num_heads,
                "attn_reduction_ratio": self.attn_reduction_ratio,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class FFNAttentionBlock(Layer):
    def __init__(
        self,
        units: int,
        num_heads: int = 1,
        mlp_ratio: int = 8,
        attn_reduction_ratio: int = 8,
        stochastic_depth_rate: float = 0.1,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.units = units
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_reduction_ratio = attn_reduction_ratio
        self.stochastic_depth_rate = stochastic_depth_rate

    def build(self, input_shape) -> None:

        self.attn = EfficientSelfAttention(
            units=self.units,
            num_heads=self.num_heads,
            attn_reduction_ratio=self.attn_reduction_ratio,
        )

        self.stochastic_depth = StochasticDepth(
            survival_probability=1 - self.stochastic_depth_rate,
        )

        self.mlp = MixFFN(
            fc1_units=self.units * self.mlp_ratio,
            fc2_units=self.units,
        )

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap1 = self.norm1(inputs)
        fmap1 = self.attn(fmap1)
        fmap1 = self.stochastic_depth([inputs, fmap1])

        fmap2 = self.norm2(fmap1)
        fmap2 = self.mlp(fmap2)

        return self.stochastic_depth([fmap1, fmap2])

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "attn_reduction_ratio": self.attn_reduction_ratio,
                "stochastic_depth_rate": self.stochastic_depth_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class SquareReshape(tf.keras.layers.Layer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

    def build(self, input_shape) -> None:

        _, tensors, _ = input_shape

        height = int(tf.sqrt(float(tensors)))
        width = int(tf.sqrt(float(tensors)))

        self.square_reshape = Reshape(target_shape=(height, width, -1))

    def call(self, inputs, training=None) -> tf.Tensor:

        return self.square_reshape(inputs)

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MixTransformer(TFModel):
    def __init__(
        self,
        img_shape: List[int],
        patch_size: List[int],
        strides: List[int],
        emb_dims: List[int],
        num_heads: List[int],
        mlp_ratios: List[int],
        stochastic_depth_rate: float,
        attn_reduction_ratios: List[int],
        depths: List[int],
        name: str,
    ) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.strides = strides
        self.emb_dims = emb_dims
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.stochastic_depth_rate = stochastic_depth_rate
        self.attn_reduction_ratios = attn_reduction_ratios
        self.depths = depths
        self.name = name

        self.endpoint_layers = [
            "reshape_stage1",
            "reshape_stage2",
            "reshape_stage3",
            "reshape_stage4",
        ]

    def get_classification_backbone(self) -> Model:
        dpr = [
            rates
            for rates in np.linspace(0, self.stochastic_depth_rate, np.sum(self.depths))
        ]

        cur = 0
        stage1 = [
            FFNAttentionBlock(
                units=self.emb_dims[0],
                num_heads=self.num_heads[0],
                mlp_ratio=self.mlp_ratios[0],
                attn_reduction_ratio=self.attn_reduction_ratios[0],
                stochastic_depth_rate=dpr[cur + idx0],
                name=f"block_{idx0}_stage_1",
            )
            for idx0 in range(self.depths[0])
        ]

        cur += self.depths[0]
        stage2 = [
            FFNAttentionBlock(
                units=self.emb_dims[1],
                num_heads=self.num_heads[1],
                mlp_ratio=self.mlp_ratios[1],
                attn_reduction_ratio=self.attn_reduction_ratios[1],
                stochastic_depth_rate=dpr[cur + idx1],
                name=f"block_{idx1}_stage_2",
            )
            for idx1 in range(self.depths[1])
        ]

        cur += self.depths[1]
        stage3 = [
            FFNAttentionBlock(
                units=self.emb_dims[2],
                num_heads=self.num_heads[2],
                mlp_ratio=self.mlp_ratios[2],
                attn_reduction_ratio=self.attn_reduction_ratios[2],
                stochastic_depth_rate=dpr[cur + idx2],
                name=f"block_{idx2}_stage_3",
            )
            for idx2 in range(self.depths[2])
        ]

        cur += self.depths[2]
        stage4 = [
            FFNAttentionBlock(
                units=self.emb_dims[3],
                num_heads=self.num_heads[3],
                mlp_ratio=self.mlp_ratios[3],
                attn_reduction_ratio=self.attn_reduction_ratios[3],
                stochastic_depth_rate=dpr[cur + idx3],
                name=f"block_{idx3}_stage_4",
            )
            for idx3 in range(self.depths[3])
        ]

        return Sequential(
            [
                Input(self.img_shape),
                OverlapPatchEmbed(
                    patch_size=self.patch_size[0],
                    strides=self.strides[0],
                    emb_dim=self.emb_dims[0],
                ),
                *stage1,
                LayerNormalization(),
                SquareReshape(name="reshape_stage1"),
                OverlapPatchEmbed(
                    patch_size=self.patch_size[1],
                    strides=self.strides[1],
                    emb_dim=self.emb_dims[1],
                ),
                *stage2,
                LayerNormalization(),
                SquareReshape(name="reshape_stage2"),
                OverlapPatchEmbed(
                    patch_size=self.patch_size[2],
                    strides=self.strides[2],
                    emb_dim=self.emb_dims[2],
                ),
                *stage3,
                LayerNormalization(),
                SquareReshape(name="reshape_stage3"),
                OverlapPatchEmbed(
                    patch_size=self.patch_size[3],
                    strides=self.strides[3],
                    emb_dim=self.emb_dims[3],
                ),
                *stage4,
                LayerNormalization(),
                SquareReshape(name="reshape_stage4"),
            ],
            name=self.name,
        )

    def get_segmentation_backbone(self) -> Model:
        backbone = self.get_classification_backbone()

        os4_output, os8_output, os16_output, os32_output = [
            backbone.get_layer(layer_name).output for layer_name in self.endpoint_layers
        ]

        return Model(
            inputs=[backbone.input],
            outputs=[os4_output, os8_output, os16_output, os32_output],
            name=self.name,
        )
