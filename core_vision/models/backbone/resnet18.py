from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Layer,
    MaxPool2D,
    ReLU,
)
from tensorflow.keras.models import Model, Sequential


@tf.keras.utils.register_keras_serializable()
class ResNetBlock(Layer):
    def __init__(
        self,
        filters: int,
        downsample: bool = False,
        l2_regul: float = 1e-4,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.downsample = downsample
        self.l2_regul = l2_regul

        self.act = ReLU()

    def build(self, input_shape) -> None:

        self.conv1 = Conv2D(
            filters=self.filters,
            strides=2,
            kernel_size=3,
            padding="same",
            use_bias=True,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=self.filters,
            strides=1,
            kernel_size=3,
            padding="same",
            use_bias=True,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.bn2 = BatchNormalization()

        if self.downsample:
            self.res_conv = Conv2D(
                self.filters,
                strides=2,
                kernel_size=1,
                kernel_initializer="he_uniform",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            )
            self.res_bn = BatchNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        skip = inputs

        fmap = self.conv1(inputs)
        fmap = self.bn1(fmap)
        fmap = self.act(fmap)
        fmap = self.conv2(fmap)
        fmap = self.bn2(fmap)

        if self.downsample:
            skip = self.res_conv(skip)
            skip = self.res_bn(skip)

        fmap = Add()([fmap, skip])
        return self.act(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"filters": self.filters, "downsample": self.downsample},
        )
        return config

    @classmethod
    def from_config(cls, config) -> None:
        return cls(**config)
