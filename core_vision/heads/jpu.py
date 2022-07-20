from typing import Any, Dict, List, Union

import tensorflow as tf
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Layer, UpSampling2D
from tensorflow.keras.models import Model, Sequential

from core_vision.layers.aspp import ASPP
from core_vision.layers.common_layers import ConvGNReLU
from core_vision.layers.joint_pyramid_upsampling import JointPyramidUpsampling


@tf.keras.utils.register_keras_serializable()
class DecoderAddon(Layer):
    def __init__(self, filters: int = 128, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters

        self.upsample2 = UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.concat = Concatenate(axis=-1)

    def build(self, input_shape) -> None:

        self.conv1 = ConvGNReLU(filters=self.filters, kernel_size=3)
        self.conv2 = ConvGNReLU(filters=self.filters, kernel_size=3)

    def call(self, inputs: List[tf.Tensor], training=None) -> tf.Tensor:

        os8_output, os4_output = inputs
        os8_fmap = self.upsample2(os8_output)
        os4_fmap = self.conv1(os4_output)

        fmap = self.concat([os8_fmap, os4_fmap])

        return self.conv2(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"filters": self.filters},
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class JPU(Layer):
    def __init__(
        self,
        n_classes: int,
        filters: int = 128,
        l2_regul: float = 1e-4,
        decoder: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(name="JPU", *args, **kwargs)
        self.n_classes = n_classes
        self.filters = filters
        self.l2_regul = l2_regul
        self.decoder = decoder

        self.scale = 4 if self.decoder else 8

        self.softmax = Activation("softmax")
        self.os4_addon = DecoderAddon(filters=self.filters)
        self.upsample = UpSampling2D(
            size=(self.scale, self.scale),
            interpolation="bilinear",
        )

    def build(self, input_shape) -> None:

        conv_config = {
            "padding": "same",
            "use_bias": True,
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }

        # JPU Module
        self.jpu = JointPyramidUpsampling()
        # fmap is of OS 8

        # ASPP Head
        self.aspp = ASPP(filters=self.filters)

        self.conv = Conv2D(
            self.n_classes,
            (3, 3),
            **conv_config,
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        os4_output, os8_output, os16_output, os32_output = inputs

        fmap = self.jpu([os8_output, os16_output, os32_output])
        fmap = self.aspp(fmap)

        if self.decoder:
            fmap = self.os4_addon([fmap, os4_output])

        fmap = self.conv(fmap)
        fmap = self.softmax(fmap)

        return self.upsample(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_classes": self.n_classes,
                "filters": self.filters,
                "decoder": self.decoder,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# def upsampling(
#     fmap: tf.Tensor,
#     height: Union[int, float],
#     width: Union[int, float],
# ) -> tf.Tensor:
#     """Upsampling module.

#     Upsample features maps to the original height, width of the images in the dataset.

#     Get the height, width of the input feature map, and the height width of the original
#     images in the dataset to compute the scale to upsample the feature map.

#     Args:
#         fmap (tf.Tensor): Input feature map of the module.
#         height (Union[int, float]): Height of the images in the dataset.
#         width (Union[int, float]): Width of the images in the dataset.

#     Returns:
#         Output feature map, size $(H,W,C)$.
#     """

#     h_fmap, w_fmap = fmap.shape.as_list()[1:3]
#     scale = (int(height // h_fmap), int(width // w_fmap))

#     return UpSampling2D(size=scale, interpolation="bilinear")(fmap)


# def decoder(
#     os8_fmap: tf.Tensor,
#     os4_fmap: tf.Tensor,
#     img_height: int,
#     img_width: int,
#     filters: int,
# ) -> tf.Tensor:
#     """Decoder part of the segmentation model.

#     Args:
#         fmap_aspp (tf.Tensor): Input feature map coming from the ASPP module.
#         endpoint (tf.Tensor): Input feature map coming from the backbone model, OS4.
#         img_height (int): Height of the images in the dataset.
#         img_width (int): Width of the images in the dataset.
#         filters (int): Number of filters used in each `conv_gn_relu` layers.

#     Returns:
#         Output feature map.
#     """

#     fmap_a = UpSampling2D(size=(2, 2), interpolation="bilinear")(os8_fmap)
#     # upsampling(fmap_aspp, img_height / 4, img_width / 4)

#     fmap_b = conv_gn_relu(os4_fmap, filters=filters, kernel_size=1)

#     fmap = Concatenate(axis=-1)([fmap_a, fmap_b])

#     return conv_gn_relu(fmap, filters=filters, kernel_size=3)


# def get_segmentation_module(
#     n_classes: int,
#     img_shape: List[int],
#     backbone: Model,
#     name: str,
# ) -> Model:
#     """Instantiate the segmentation head module for the segmentation task.

#     Args:
#         n_classes (int): Number of classes in the segmentation task.
#         img_shape (List[int]): Input shape of the images/masks in the dataset.
#         backbone (Model): CNN used as backbone/feature extractor.
#         name (str): Name of the segmentation head module.

#     Returns:
#         A semantic segmentation model.
#     """

#     img_height, img_width = img_shape[:2]

#     os4_output, os8_output, os16_output, os32_output = backbone.outputs

#     # JPU Module
#     fmap = JointPyramidUpsampling()([os8_output, os16_output, os32_output])
#     # fmap is of OS 8

#     # ASPP Head
#     fmap = ASPP(filters=128)(fmap)
#     # fmap is of OS 8

#     fmap = decoder(fmap, os4_output, img_height, img_width, 128)

#     fmap = Conv2D(
#         n_classes,
#         (3, 3),
#         activation="softmax",
#         padding="same",
#         name="output_layer",
#     )(fmap)
#     out = upsampling(fmap, img_height, img_width)

#     return Model(inputs=backbone.input, outputs=out, name=name)
