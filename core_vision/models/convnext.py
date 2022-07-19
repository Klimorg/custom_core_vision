from typing import Any, Dict, List

import tensorflow as tf
from loguru import logger
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    Input,
    Layer,
    LayerNormalization,
)
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable()
class ConvNextBlock(Layer):
    def __init__(
        self,
        filters: int,
        expansion_rate: int = 4,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.expansion_rate = expansion_rate
        self.l2_regul = l2_regul

        self.gelu = gelu

        self.layer_norm = LayerNormalization()

    def build(self, input_shape) -> None:

        self.dw_conv = DepthwiseConv2D(
            kernel_size=7,
            strides=1,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_uniform",
            depthwise_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.inverted_bottleneck = Conv2D(
            filters=self.expansion_rate * self.filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = self.dw_conv(inputs)
        fmap = self.layer_norm(fmap)
        fmap = self.inverted_bottleneck(fmap)
        fmap = self.gelu(fmap)
        fmap = self.conv(fmap)

        return fmap + inputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "expansion_rate": self.expansion_rate,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ConvNeXtLayer(Layer):
    def __init__(self, filters: int, num_blocks: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.num_blocks = num_blocks

    def build(self, input_shape) -> None:

        self.blocks = [
            ConvNextBlock(filters=self.filters) for _ in range(self.num_blocks)
        ]

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = inputs

        for block in self.blocks:
            fmap = block(fmap)

        return fmap

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"filters": self.filters, "num_blocks": self.num_blocks},
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConvNext(Model):
    def __init__(
        self,
        filters: List[int],
        num_blocks: List[int],
        name: str,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(name=name, *args, **kwargs)

        conv_config = {
            "padding": "same",
            "use_bias": True,
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=1e-4),
        }

        self.stem = Conv2D(
            filters=filters[0],
            kernel_size=4,
            strides=4,
            **conv_config,
            name="stem",
        )
        self.stem_ln = LayerNormalization(name="stem_layer_norm")

        self.convnext_l1 = ConvNeXtLayer(
            filters=filters[0],
            num_blocks=num_blocks[0],
            name="convnext_layer_1",
        )
        self.ln_l1 = LayerNormalization(name="downsample_1_layer_norm")
        self.down_l1 = Conv2D(
            filters=filters[1],
            kernel_size=2,
            strides=2,
            **conv_config,
            name="downsample_1",
        )

        self.convnext_l2 = ConvNeXtLayer(
            filters=filters[1],
            num_blocks=num_blocks[1],
            name="convnext_layer_2",
        )
        self.ln_l2 = LayerNormalization(name="downsample_2_layer_norm")
        self.down_l2 = Conv2D(
            filters=filters[2],
            kernel_size=2,
            strides=2,
            **conv_config,
            name="downsample_2",
        )

        self.convnext_l3 = ConvNeXtLayer(
            filters=filters[2],
            num_blocks=num_blocks[2],
            name="convnext_layer_3",
        )
        self.ln_l3 = LayerNormalization(name="downsample_3_layer_norm")
        self.down_l3 = Conv2D(
            filters=filters[3],
            kernel_size=2,
            strides=2,
            **conv_config,
            name="downsample_3",
        )

        self.convnext_l4 = ConvNeXtLayer(
            filters=filters[3],
            num_blocks=num_blocks[3],
            name="convnext_layer_4",
        )

    def call(self, inputs) -> tf.Tensor:

        fmap = self.stem(inputs)

        fmap = self.convnext_l1(fmap)
        fmap = self.ln_l1(fmap)
        fmap = self.down_l1(fmap)

        fmap = self.convnext_l2(fmap)
        fmap = self.ln_l2(fmap)
        fmap = self.down_l2(fmap)

        fmap = self.convnext_l3(fmap)
        fmap = self.ln_l3(fmap)
        fmap = self.down_l3(fmap)

        return self.convnext_l4(fmap)
