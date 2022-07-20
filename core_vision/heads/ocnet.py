from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Layer, UpSampling2D
from tensorflow.keras.models import Model

from core_vision.layers.common_layers import ConvBNReLU
from core_vision.layers.object_context import ASPP_OC, BaseOC


@tf.keras.utils.register_keras_serializable()
class BaseOC(Layer):
    def __init__(
        self,
        filters: int,
        n_classes: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.n_classes = n_classes
        self.l2_regul = l2_regul

        self.softmax = Activation("softmax")

        self.upsample8 = UpSampling2D(size=(8, 8), interpolation="bilinear")

    def build(self, input_shape) -> None:

        conv_config = {
            "padding": "same",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
            "kernel_initializer": "he_uniform",
            "use_bias": True,
        }
        self.conv1 = ConvBNReLU(filters=1024, kernel_size=3)
        self.base_oc = BaseOC(filters=self.filters)

        self.conv2 = Conv2D(
            filters=self.n_classes,
            kernel_size=(3, 3),
            **conv_config,
        )

    def call(self, inputs: List[tf.Tensor], training=None) -> tf.Tensor:

        _, os8_output, *_ = inputs

        fmap = self.conv1(os8_output)
        fmap = self.base_oc(fmap)
        fmap = self.conv2(fmap)
        fmap = self.softmax(fmap)

        return self.upsample8(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "n_classes": self.n_classes,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AsppOC(Layer):
    def __init__(
        self,
        filters: int,
        n_classes: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.n_classes = n_classes
        self.l2_regul = l2_regul

        self.softmax = Activation("softmax")

        self.upsample8 = UpSampling2D(size=(8, 8), interpolation="bilinear")

    def build(self, input_shape) -> None:

        self.aspp = ASPP_OC(filters=self.filters)

        conv_config = {
            "padding": "same",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
            "kernel_initializer": "he_uniform",
            "use_bias": True,
        }

        self.conv = Conv2D(
            filters=self.n_classes,
            kernel_size=(3, 3),
            **conv_config,
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        _, os8_output, *_ = inputs

        fmap = self.aspp(os8_output)
        fmap = self.conv(fmap)
        fmap = self.softmax(fmap)

        return self.upsample8(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "n_classes": self.n_classes,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# def get_segmentation_module(
#     n_classes: int,
#     backbone: Model,
#     architecture: str,
#     filters: int,
#     name: str,
# ) -> Model:
#     """Instantiate the segmentation head module for the segmentation task.

#     Args:
#         n_classes (int): Number of classes in the segmentation task.
#         backbone (tf.keras.Model): CNN used as backbone/feature extractor.
#         architecture (str): Choice of architecture for the segmentation head : `base_oc` or `aspp_ocnet`.
#         filters (int): Numbers of filters used in the segmentation head.
#         name (str): Name of the segmentation head module.

#     Returns:
#         A semantic segmentation model.
#     """

#     fmap = backbone.outputs[1]

#     if architecture == "base_oc":
#         fmap = conv_bn_relu(fmap, filters=1024, kernel_size=3, name="pre_OCP_conv")
#         fmap = BaseOC(filters=filters)(fmap)
#     elif architecture == "aspp_ocnet":
#         fmap = ASPP_OC(filters=filters)(fmap)

#     fmap = Conv2D(
#         filters=n_classes,
#         kernel_size=(3, 3),
#         padding="same",
#         kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
#         kernel_initializer="he_uniform",
#         use_bias=False,
#     )(fmap)

#     fmap = Activation("softmax")(fmap)

#     out = UpSampling2D(size=(8, 8), interpolation="bilinear")(fmap)

#     return Model(inputs=[backbone.inputs], outputs=[out], name=name)
