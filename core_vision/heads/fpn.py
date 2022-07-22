from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv2D, Layer, UpSampling2D

from core_vision.layers.feature_pyramids import FeaturePyramidNetwork, SemanticHeadFPN


@tf.keras.utils.register_keras_serializable()
class FPN(Layer):
    def __init__(self, n_classes: int, l2_regul: float = 1e-4, *args, **kwargs) -> None:
        super().__init__(name="FPN", *args, **kwargs)

        self.n_classes = n_classes
        self.l2_regul = l2_regul

        self.upsample4 = UpSampling2D(size=(4, 4), interpolation="bilinear")
        self.softmax = Activation("softmax")

    def build(self, input_shape) -> None:

        conv_config = {
            "padding": "same",
            "use_bias": True,
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }
        self.fpn = FeaturePyramidNetwork()

        self.head = SemanticHeadFPN()

        self.conv = Conv2D(
            filters=self.n_classes,
            kernel_size=(1, 1),
            strides=(1, 1),
            **conv_config,
        )

    def call(self, inputs: List[tf.Tensor], training=None) -> tf.Tensor:

        # os4_output, os8_output, os16_output, os32 = inputs

        fmap = self.fpn(inputs)

        fmap = self.head(fmap)

        fmap = self.conv(fmap)

        fmap = self.upsample4(fmap)
        return self.softmax(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
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
#     name: str,
# ) -> Model:
#     """Instantiate the segmentation head module for the segmentation task.

#     Args:
#         n_classes (int): Number of classes in the segmentation task.
#         backbone (tf.keras.Model): CNN used as backbone/feature extractor.
#         name (str): Name of the segmentation head module.

#     Returns:
#         A semantic segmentation model.
#     """

#     c_outputs = backbone.outputs

#     p_outputs = FeaturePyramidNetwork()(c_outputs)

#     fmap = SemanticHeadFPN()(p_outputs)

#     fmap = Conv2D(
#         filters=n_classes,
#         kernel_size=(1, 1),
#         strides=(1, 1),
#         padding="same",
#         kernel_initializer="he_uniform",
#     )(fmap)

#     fmap = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap)
#     out = Activation("softmax")(fmap)

#     return Model(inputs=[backbone.inputs], outputs=[out], name=name)
