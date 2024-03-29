from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Input,
    Layer,
    LayerNormalization,
    MaxPool2D,
    ReLU,
)
from tensorflow.keras.models import Model, Sequential

from core_vision.models.utils import TFModel


@tf.keras.utils.register_keras_serializable()
class ConvTokenizer(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int = 64,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul

    def build(self, input_shape) -> None:

        conv_config = {
            "padding": "same",
            "use_bias": False,
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }

        self.block = Sequential(
            [
                Conv2D(
                    self.filters // 2,
                    kernel_size=3,
                    strides=2,
                    **conv_config,
                ),
                BatchNormalization(),
                ReLU(),
                Conv2D(
                    self.filters // 2,
                    kernel_size=3,
                    strides=1,
                    **conv_config,
                ),
                BatchNormalization(),
                ReLU(),
                Conv2D(
                    self.filters,
                    kernel_size=3,
                    strides=1,
                    **conv_config,
                ),
                BatchNormalization(),
                ReLU(),
                MaxPool2D(pool_size=3, strides=2, padding="same"),
            ],
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        return self.block(inputs)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "l2_regularization": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ConvStage(tf.keras.layers.Layer):
    def __init__(
        self,
        num_blocks: int = 2,
        filters_in: int = 128,
        filters_out: int = 64,
        filters_downsample: int = 64,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.filters_downsample = filters_downsample
        self.l2_regul = l2_regul

    def build(self, input_shape) -> None:

        conv_config = {
            "padding": "same",
            "use_bias": False,
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }

        self.conv_blocks = [
            Sequential(
                [
                    Conv2D(
                        self.filters_in,
                        kernel_size=1,
                        strides=1,
                        **conv_config,
                    ),
                    BatchNormalization(),
                    ReLU(),
                    Conv2D(
                        self.filters_in,
                        kernel_size=3,
                        strides=1,
                        **conv_config,
                    ),
                    BatchNormalization(),
                    ReLU(),
                    Conv2D(
                        self.filters_out,
                        kernel_size=1,
                        strides=1,
                        **conv_config,
                    ),
                    BatchNormalization(),
                    ReLU(),
                ],
            )
            for _ in range(self.num_blocks)
        ]

        self.downsample = Conv2D(
            self.filters_downsample,
            kernel_size=3,
            strides=2,
            **conv_config,
        )

    def call(self, inputs, trainable=None) -> tf.Tensor:
        fmap = inputs
        for block in self.conv_blocks:
            fmap = fmap + block(fmap)

        return self.downsample(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "filters_in": self.filters_in,
                "filters_out": self.filters_out,
                "filters_downsample": self.filters_downsample,
                "l2_regularization": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ConvDownsample(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul

    def build(self, input_shape) -> None:

        conv_config = {
            "padding": "same",
            "use_bias": False,
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }

        self.downsample = Conv2D(
            self.filters,
            kernel_size=3,
            strides=2,
            **conv_config,
        )

    def call(self, inputs) -> tf.Tensor:

        return self.downsample(inputs)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "l2_regularization": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Referred from: github.com:rwightman/pytorch-image-models.
# https://keras.io/examples/vision/cct/#stochastic-depth-for-regularization
@tf.keras.utils.register_keras_serializable()
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(
        self,
        drop_prop,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.drop_prob = drop_prop

    def call(self, inputs, training=None) -> tf.Tensor:
        if training:
            keep_prob = tf.cast(1 - self.drop_prob, dtype=inputs.dtype)
            shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
            random_tensor = keep_prob + tf.random.uniform(
                shape,
                0,
                1,
                dtype=inputs.dtype,
            )
            random_tensor = tf.floor(random_tensor)
            return (inputs / keep_prob) * random_tensor
        return inputs

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


@tf.keras.utils.register_keras_serializable()
class Identity(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__(name="IdentityTF")

    def call(self, inputs) -> tf.Tensor:
        return inputs


@tf.keras.utils.register_keras_serializable()
class Mlp(tf.keras.layers.Layer):
    def __init__(
        self,
        fc1_units: int,
        fc2_units: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.l2_regul = l2_regul

        self.gelu = tf.keras.activations.gelu

    def build(self, input_shape) -> None:

        self.fc1 = Dense(
            units=self.fc1_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.fc2 = Dense(
            units=self.fc2_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = self.fc1(inputs)
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


@tf.keras.utils.register_keras_serializable()
class ConvMLPStage(tf.keras.layers.Layer):
    def __init__(
        self,
        expansion_units: int,
        units: int,
        stochastic_depth_rate: float = 0.1,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.expansion_units = expansion_units
        self.units = units
        self.stochastic_depth_rate = stochastic_depth_rate
        self.l2_regul = l2_regul

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.connect_norm = LayerNormalization()

    def build(self, input_shape) -> None:

        self.channel_mlp1 = Mlp(fc1_units=self.expansion_units, fc2_units=self.units)
        self.channel_mlp2 = Mlp(fc1_units=self.expansion_units, fc2_units=self.units)
        self.stochastic_drop: Layer = (
            StochasticDepth(drop_prop=self.stochastic_depth_rate)
            if self.stochastic_depth_rate > 0
            else Identity()
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

        fmap = inputs + self.stochastic_drop(self.channel_mlp1(self.norm1(inputs)))
        fmap = self.depth_conv(self.connect_norm(fmap))
        return fmap + self.stochastic_drop(self.channel_mlp2(self.norm2(inputs)))

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "expansion_units": self.expansion_units,
                "units": self.units,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "l2_regularization": self.l2_regul,
            },
        )
        return config


@tf.keras.utils.register_keras_serializable()
class BasicStage(tf.keras.layers.Layer):
    def __init__(
        self,
        num_blocks: int,
        units: int,
        mlp_ratio: int = 1,
        stochastic_depth_rate: float = 0.1,
        downsample: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks
        self.units = units
        self.mlp_ratio = mlp_ratio
        self.stochastic_depth_rate = stochastic_depth_rate
        self.downsample = downsample

    def build(self, input_shape) -> None:

        dpr = [
            rates
            for rates in np.linspace(0, self.stochastic_depth_rate, self.num_blocks)
        ]

        self.blocks = [
            ConvMLPStage(
                expansion_units=int(self.units * self.mlp_ratio),
                units=self.units,
                stochastic_depth_rate=dpr[idx],
            )
            for idx in range(self.num_blocks)
        ]

        self.downsample_mlp: Layer = (
            ConvDownsample(filters=int(self.units * 2))
            if self.downsample
            else Identity()
        )

    def call(self, inputs, trainable=None) -> tf.Tensor:

        for blk in self.blocks:
            inputs = blk(inputs)

        return self.downsample_mlp(inputs)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "units": self.units,
                "mlp_ratio": self.mlp_ratio,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "downsample": self.downsample,
            },
        )
        return config


class ConvMLP(TFModel):
    def __init__(
        self,
        img_shape: List[int],
        channels: int,
        n_conv_blocks: int,
        num_blocks: List[int],
        units: List[int],
        mlp_ratios: List[int],
        name: str,
    ) -> None:
        super().__init__()

        self.img_shape = img_shape
        self.channels = channels
        self.n_conv_blocks = n_conv_blocks
        self.num_blocks = num_blocks
        self.units = units
        self.mlp_ratios = mlp_ratios
        self.name = name

        self.endpoint_layers = [
            "tokenizer",
            "conv",
            "mlp1",
            "mlp3",
        ]

    def get_classification_backbone(self) -> Model:
        return Sequential(
            [
                Input(self.img_shape),
                ConvTokenizer(filters=self.channels, name="tokenizer"),
                ConvStage(
                    num_blocks=self.n_conv_blocks,
                    filters_out=self.channels,
                    filters_downsample=self.units[0],
                    name="conv",
                ),
                BasicStage(
                    num_blocks=self.num_blocks[0],
                    units=self.units[1],
                    mlp_ratio=self.mlp_ratios[0],
                    downsample=True,
                    name="mlp1",
                ),
                BasicStage(
                    num_blocks=self.num_blocks[1],
                    units=self.units[2],
                    mlp_ratio=self.mlp_ratios[1],
                    downsample=True,
                    name="mlp2",
                ),
                BasicStage(
                    num_blocks=self.num_blocks[2],
                    units=self.units[3],
                    mlp_ratio=self.mlp_ratios[2],
                    downsample=False,
                    name="mlp3",
                ),
            ],
            name=self.name,
        )

    def get_segmentation_backbone(
        self,
    ) -> Model:
        """Instantiate the model and use it as a backbone (feature extractor) for a semantic segmentation task.

        Returns:
            A `tf.keras` model.
        """

        backbone = self.get_classification_backbone()

        os4_output, os8_output, os16_output, os32_output = [
            backbone.get_layer(layer_name).output for layer_name in self.endpoint_layers
        ]

        return Model(
            inputs=[backbone.input],
            outputs=[os4_output, os8_output, os16_output, os32_output],
            name=self.name,
        )
