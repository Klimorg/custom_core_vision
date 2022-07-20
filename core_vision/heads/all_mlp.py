from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Concatenate, Dense, Layer, UpSampling2D


@tf.keras.utils.register_keras_serializable()
class AllMLP(Layer):
    def __init__(
        self,
        units: int,
        n_classes: int,
        l2_regul: float = 1e-4,
        interpolation: str = "bilinear",
        *args,
        **kwargs,
    ) -> None:

        super().__init__(name="AllMLP", *args, **kwargs)
        self.units = units
        self.n_classes = n_classes
        self.l2_regul = l2_regul
        self.interpolation = interpolation

        self.softmax = Activation("softmax")
        self.concat = Concatenate(axis=-1)

        self.upsample2 = UpSampling2D(size=(2, 2), interpolation=self.interpolation)
        self.upsample4 = UpSampling2D(size=(4, 4), interpolation=self.interpolation)
        self.upsample8 = UpSampling2D(size=(8, 8), interpolation=self.interpolation)

    def build(self, input_shape) -> None:

        dense_config = {
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }

        self.dense1 = Dense(
            self.units,
            **dense_config,
        )

        self.dense2 = Dense(
            self.units,
            **dense_config,
        )

        self.dense3 = Dense(
            self.units,
            **dense_config,
        )

        self.dense4 = Dense(
            self.units,
            **dense_config,
        )

        self.dense5 = Dense(
            self.units,
            **dense_config,
        )

        self.dense6 = Dense(
            self.n_classes,
            **dense_config,
        )

    def call(self, inputs: List[tf.Tensor], training=None) -> tf.Tensor:

        os4_output, os8_output, os16_output, os32_output = inputs

        fmap1 = self.dense1(os4_output)

        fmap2 = self.dense2(os8_output)
        fmap2 = self.upsample2(fmap2)

        fmap3 = self.dense3(os16_output)
        fmap3 = self.upsample4(fmap3)

        fmap4 = self.dense4(os32_output)
        fmap4 = self.upsample8(fmap4)

        fmap = self.concat([fmap1, fmap2, fmap3, fmap4])
        fmap = self.dense5(fmap)
        fmap = self.dense6(fmap)
        fmap = self.upsample4(fmap)

        return self.softmax(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "n_classes": self.n_classes,
                "l2_regul": self.l2_regul,
                "interpolation": self.interpolation,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# def get_segmentation_module(
#     units: int,
#     n_classes: int,
#     backbone: Model,
#     name: str,
# ) -> Model:
#     """Instantiate the segmentation head module for the segmentation task.

#     Args:
#         n_classes (int): Number of classes in the segmentation task.
#         backbone (Model): CNN used as backbone/feature extractor.
#         name (str): Name of the segmentation head module.

#     Returns:
#         A semantic segmentation model.
#     """

#     l2_regul = 1e-4
#     bil = "bilinear"
#     he = "he_uniform"

#     os4_output, os8_output, os16_output, os32_output = backbone.outputs

#     fmap1 = Dense(
#         units,
#         kernel_initializer=he,
#         kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
#     )(os4_output)

#     fmap2 = Dense(
#         units,
#         kernel_initializer=he,
#         kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
#     )(os8_output)
#     fmap2 = UpSampling2D(size=(2, 2), interpolation=bil)(fmap2)

#     fmap3 = Dense(
#         units,
#         kernel_initializer=he,
#         kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
#     )(os16_output)
#     fmap3 = UpSampling2D(size=(4, 4), interpolation=bil)(fmap3)

#     fmap4 = Dense(
#         units,
#         kernel_initializer=he,
#         kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
#     )(os32_output)
#     fmap4 = UpSampling2D(size=(8, 8), interpolation=bil)(fmap4)

#     fmap = Concatenate(axis=-1)([fmap1, fmap2, fmap3, fmap4])

#     fmap = Dense(
#         units,
#         kernel_initializer=he,
#         kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
#     )(fmap)

#     fmap = Dense(
#         n_classes,
#         kernel_initializer=he,
#         kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
#     )(fmap)

#     fmap = UpSampling2D(size=(4, 4), interpolation=bil)(fmap)
#     out = Activation("softmax")(fmap)

#     return Model(inputs=[backbone.inputs], outputs=[out], name=name)
