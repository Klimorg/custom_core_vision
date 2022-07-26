from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input

from core_vision.layers.common_layers import InvertedResidualBottleneck2D


def get_feature_extractor(
    img_shape: List[int],
) -> tf.keras.Model:
    """Instantiate a Mobilenetv2 model.

    Args:
        img_shape (List[int]): Input shape of the images in the dataset.

    Returns:
        A `tf.keras` model.
    """
    channels = [32, 16, 24, 32, 64, 96, 160, 320]

    img_input = Input(img_shape)

    img = Conv2D(
        filters=channels[0],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=True,
    )(img_input)

    img = InvertedResidualBottleneck2D(
        expansion_rate=1,
        filters=channels[1],
        strides=1,
        skip_connection=False,
        name="inv_bottleneck_1",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[2],
        strides=2,
        skip_connection=False,
        name="inv_bottleneck_2_1",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[2],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_2_2",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[3],
        strides=2,
        skip_connection=False,
        name="inv_bottleneck_3_1",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[3],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_3_2",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[3],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_3_3",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[4],
        strides=2,
        skip_connection=False,
        name="inv_bottleneck_4_1",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[4],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_4_2",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[4],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_4_3",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[4],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_4_4",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[5],
        strides=1,
        skip_connection=False,
        name="inv_bottleneck_5_1",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[5],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_5_2",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[5],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_5_3",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[6],
        strides=2,
        skip_connection=False,
        name="inv_bottleneck_6_1",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[6],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_6_2",
    )(img)

    img = InvertedResidualBottleneck2D(
        expansion_rate=6,
        filters=channels[6],
        strides=1,
        skip_connection=True,
        name="inv_bottleneck_6_3",
    )(img)

    img = Conv2D(
        filters=1280,
        kernel_size=1,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=True,
    )(img)

    return Model(img_input, img)


def get_backbone(img_shape: List[int], backbone_name: str) -> tf.keras.Model:
    """Instantiate the model and use it as a backbone (feature extractor) for a semantic segmentation task.

    Args:
        img_shape (List[int]): Input shape of the images/masks in the dataset.
        backbone_name (str): Name of the backbone.

    Returns:
        A `tf.keras` model.
    """

    backbone = get_feature_extractor(
        img_shape=img_shape,
    )

    endpoint_layers = [
        "inv_bottleneck_2_2",
        "inv_bottleneck_3_3",
        "inv_bottleneck_5_3",
        "inv_bottleneck_6_3",
    ]

    os4_output, os8_output, os16_output, os32_output = [
        backbone.get_layer(layer_name).output for layer_name in endpoint_layers
    ]

    return Model(
        inputs=[backbone.input],
        outputs=[os4_output, os8_output, os16_output, os32_output],
        name=backbone_name,
    )
