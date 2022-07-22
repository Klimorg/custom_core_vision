from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Concatenate, Input, Layer, MaxPool2D

from core_vision.layers.common_layers import ConvBNReLU
from core_vision.models.utils import TFModel


@tf.keras.utils.register_keras_serializable()
class OSAModule(Layer):
    def __init__(
        self,
        filters_conv3x3: int,
        filters_conv1x1: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filters_conv3x3 = filters_conv3x3
        self.filters_conv1x1 = filters_conv1x1

        self.concat = Concatenate(axis=-1)

    def build(self, input_shape) -> None:

        self.conv1 = ConvBNReLU(
            filters=self.filters_conv3x3,
            kernel_size=3,
        )
        self.conv2 = ConvBNReLU(
            filters=self.filters_conv3x3,
            kernel_size=3,
        )
        self.conv3 = ConvBNReLU(
            filters=self.filters_conv3x3,
            kernel_size=3,
        )
        self.conv4 = ConvBNReLU(
            filters=self.filters_conv3x3,
            kernel_size=3,
        )
        self.conv5 = ConvBNReLU(
            filters=self.filters_conv3x3,
            kernel_size=3,
        )
        self.conv6 = ConvBNReLU(
            filters=self.filters_conv1x1,
            kernel_size=1,
        )

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        fmap1 = self.conv1(inputs)
        fmap2 = self.conv2(inputs)
        fmap3 = self.conv3(inputs)
        fmap4 = self.conv4(inputs)
        fmap5 = self.conv5(inputs)

        fmap = self.concat([fmap1, fmap2, fmap3, fmap4, fmap5])

        return self.conv6(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters_conv3x3": self.filters_conv3x3,
                "filters_conv1x1": self.filters_conv1x1,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VoVNet(TFModel):
    def __init__(
        self,
        img_shape: List[int],
        filters_conv3x3: List[int],
        filters_conv1x1: List[int],
        block_repetitions: List[int],
        name: str,
    ) -> None:

        self.img_shape = img_shape
        self.filters_conv3x3 = filters_conv3x3
        self.filters_conv1x1 = filters_conv1x1
        self.block_repetitions = block_repetitions
        self.name = name

        self.endpoint_layers = [
            "maxpool_block1_out",
            "maxpool_block2_out",
            "maxpool_block3_out",
            "maxpool_block4_out",
        ]

    def get_classification_backbone(self) -> Model:

        block1 = [
            OSAModule(
                filters_conv3x3=self.filters_conv3x3[0],
                filters_conv1x1=self.filters_conv1x1[0],
                name=f"block_1_{idx}",
            )
            for idx in range(self.block_repetitions[0])
        ]

        block2 = [
            OSAModule(
                filters_conv3x3=self.filters_conv3x3[1],
                filters_conv1x1=self.filters_conv1x1[1],
                name=f"block_2_{idx}",
            )
            for idx in range(self.block_repetitions[1])
        ]

        block3 = [
            OSAModule(
                filters_conv3x3=self.filters_conv3x3[2],
                filters_conv1x1=self.filters_conv1x1[2],
                name=f"block_3_{idx}",
            )
            for idx in range(self.block_repetitions[2])
        ]

        block4 = [
            OSAModule(
                filters_conv3x3=self.filters_conv3x3[3],
                filters_conv1x1=self.filters_conv1x1[3],
                name=f"block_4_{idx}",
            )
            for idx in range(self.block_repetitions[3])
        ]

        return Sequential(
            [
                Input(self.img_shape),
                ConvBNReLU(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    name="stem_stage_1",
                ),
                ConvBNReLU(
                    filters=64,
                    kernel_size=3,
                    name="stem_stage_2",
                ),
                ConvBNReLU(
                    filters=128,
                    kernel_size=3,
                    name="stem_stage_3",
                ),
                *block1,
                MaxPool2D(pool_size=(2, 2), name="maxpool_block1_out"),
                *block2,
                MaxPool2D(pool_size=(2, 2), name="maxpool_block2_out"),
                *block3,
                MaxPool2D(pool_size=(2, 2), name="maxpool_block3_out"),
                *block4,
                MaxPool2D(pool_size=(2, 2), name="maxpool_block4_out"),
            ],
            name=self.name,
        )

    def get_segmentation_backbone(self) -> Model:
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
