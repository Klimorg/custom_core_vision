from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Layer, UpSampling2D

from core_vision.layers.common_layers import ConvBNReLU
from core_vision.layers.shared_kernels import KSAConv2D


@tf.keras.utils.register_keras_serializable()
class KSAC(Layer):
    def __init__(
        self,
        n_classes: int,
        filters: int,
        ksac_filters: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name="KSAC", *args, **kwargs)

        self.n_classes = n_classes
        self.filters = filters
        self.ksac_filters = ksac_filters
        self.l2_regul = l2_regul

        self.upsample4 = UpSampling2D(size=(4, 4), interpolation="bilinear")
        self.concat = Concatenate(axis=-1)
        self.softmax = Activation("softmax")

    def build(self, input_shape) -> None:
        conv_config = {
            "padding": "same",
            "use_bias": True,
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }

        self.conv1 = ConvBNReLU(filters=self.filters, kernel_size=1, name="decoder1")
        self.conv2 = ConvBNReLU(filters=self.filters, kernel_size=3, name="decoder2")

        self.ksac = KSAConv2D(filters=self.ksac_filters)

        self.conv3 = Conv2D(
            filters=self.n_classes,
            kernel_size=(3, 3),
            strides=(1, 1),
            **conv_config,
        )

    def call(self, inputs: List[tf.Tensor], training=None) -> tf.Tensor:
        os4_output, _, os16_output, _ = inputs

        fmap1 = self.ksac(os16_output)
        fmap1 = self.upsample4(fmap1)

        fmap2 = self.conv1(os4_output)

        fmap = self.concat([fmap1, fmap2])
        fmap = self.conv2(fmap)
        fmap = self.upsample4(fmap)
        fmap = self.conv3(fmap)

        return self.softmax(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_classes": self.n_classes,
                "filters": self.filters,
                "ksac_filters": self.ksac_filters,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# def decoder(fmap1, fmap2, filters):
#     """[summary]

#     Args:
#         fmap1 ([type]): [description]
#         fmap2 ([type]): [description]
#         filters ([type]): [description]

#     Returns:
#         [type]: [description]
#     """

#     fmap1 = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap1)

#     fmap2 = ConvBNReLU(filters=filters, kernel_size=1, name="decoder1")(fmap2)

#     fmap = Concatenate(axis=-1)([fmap1, fmap2])

#     return ConvBNReLU(filters=filters, kernel_size=3, name="decoder2")(fmap)


# def get_segmentation_module(
#     n_classes: int,
#     backbone: Model,
#     name: str,
#     ksac_filters: int,
#     decoder_filters: int,
# ) -> Model:
#     """Instantiate the segmentation head module for the segmentation task.

#     Args:
#         n_classes (int): Number of classes in the segmentation task.
#         backbone (tf.keras.Model): CNN used as backbone/feature extractor.
#         name (str): Name of the segmentation head module.

#     Returns:
#         A semantic segmentation model.
#     """

#     c2_output, _, c4_output, _ = backbone.outputs

#     fm = KSAConv2D(filters=ksac_filters)(c4_output)

#     fmap = decoder(fm, c2_output, filters=decoder_filters)

#     fmap = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap)

#     fmap = Conv2D(
#         filters=n_classes,
#         kernel_size=(3, 3),
#         strides=(1, 1),
#         padding="same",
#         kernel_initializer="he_uniform",
#     )(fmap)

#     out = Activation("softmax")(fmap)

#     return Model(inputs=[backbone.inputs], outputs=[out], name=name)
