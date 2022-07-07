from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    GlobalAveragePooling2D,
    Layer,
)
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable()
class ClassificationHead(Layer):
    def __init__(
        self, units: int, num_classes: int, l2_regul: float = 1e-4, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.units = units
        self.num_classes = num_classes
        self.l2_regul = l2_regul

        self.act = Activation("softmax")

    def build(self, input_shape) -> None:
        batch_size, width, height, channels = input_shape
        self.global_avg = GlobalAveragePooling2D()
        self.dense1 = Dense(
            units=self.units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.dense2 = Dense(
            units=self.num_classes,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = self.global_avg(inputs)
        fmap = self.dense1(fmap)
        fmap = self.dense2(fmap)

        return self.act(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_classes": self.num_classes,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config) -> None:
        return cls(**config)
