from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Input,
    Layer,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential

from core_vision.models.utils import TFModel


@tf.keras.utils.register_keras_serializable()
class SqueezeAndExcite(Layer):
    """Squeeze-and-Excitation Module.

    Architecture:
        ![architecture](./images/se_module.svg)

        Source : [ArXiv link](https://arxiv.org/abs/1709.01507)

    Args:
        fmap_in (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        ratio (int): Define the ratio of filters used in the squeeze operation of the modle (the first Conv2D).
        filters (int): Numbers of filters used in the excitation operation of the module (the second Conv2D).
        name (str): Name of the module.
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.
    Returns:
        Output feature map, size = $(H,W,C)$.
    """

    def __init__(
        self,
        ratio: int,
        filters: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.filters = filters
        self.l2_regul = l2_regul

        self.excitation = Activation("sigmoid")
        self.global_avg = GlobalAveragePooling2D()
        self.relu = ReLU()

    def build(self, input_shape) -> None:
        channels = input_shape[-1]

        conv_config = {
            "padding": "same",
            "use_bias": False,
            "kernel_initializer": "he_uniform",
            "kernel_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }

        self.reshape = Reshape((1, 1, channels))

        self.conv1 = Conv2D(
            filters=int(self.filters / self.ratio),
            kernel_size=(1, 1),
            strides=(1, 1),
            **conv_config,
        )
        self.conv2 = Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            **conv_config,
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        excite = self.global_avg(inputs)
        excite = self.reshape(excite)
        excite = self.conv1(excite)
        excite = self.relu(excite)
        excite = self.conv2(excite)
        excite = self.excitation(excite)

        return inputs * excite

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "ratio": self.ratio,
                "filters": self.filters,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class GhostModule(Layer):
    """Primary module of the GhostNet architecture.

    Args:
        fmap (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        out (int): Number of channels of the output feature map.
        ratio (int): Define the ratio between the number of filters of the Conv2D layer
            and the number of filters of the `DepthwiseConv2D` in the last `Concatenate`
            layer. `depth_multiplier` of the `DepthwiseConv2D` layer is also defined as
            `ratio-1`.
        convkernel (Tuple[int, int]): Number of convolution kernels in the `Conv2D` layer.
        dwkernel (Tuple[int, int]): Number of convolution kernels in the `DepthwiseConv2D` layer.
        name (str): Name of the module.
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.

    Returns:
        Output feature map, size = $(H,W,\mathrm{out})$
    """

    def __init__(
        self,
        out: int,
        ratio: int,
        convkernel: int,
        dwkernel: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.out = out
        self.ratio = ratio
        self.convkernel = convkernel
        self.dwkernel = dwkernel
        self.l2_regul = l2_regul

        self.filters = int(np.ceil(out / ratio))
        self.channels = int(out - self.filters)

        self.concat = Concatenate(axis=-1)

    def build(self, input_shape) -> None:

        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.convkernel,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.dwconv = DepthwiseConv2D(
            kernel_size=self.dwkernel,
            strides=(1, 1),
            depth_multiplier=self.ratio - 1,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_uniform",
            depthwise_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:

        fmap = self.conv(inputs)
        dwfmap = self.dwconv(fmap)

        return self.concat(
            [fmap, dwfmap[:, :, :, : self.channels]],
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "out": self.out,
                "ratio": self.ratio,
                "convkernel": self.convkernel,
                "dwkernel": self.dwkernel,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class GhostBottleneckModule(Layer):
    """Ghost Bottleneck Module, the backbone of the GhostNet model.

    Args:
        fmap_in (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        dwkernel (int): Number of convolution kernels in the `DepthwiseConv2D` layer.
        strides (int): Stride used in the `DepthwiseConv2D` layers.
        exp (int): Number of filters used as an expansion operation in the first `ghost_module`.
        out (int): Number of filters/channels of the output feature map.
        ratio (int): Define the ratio in the `ghost_module` between the number of filters of the Conv2D layer
            and the number of filters of the `DepthwiseConv2D` in the last `Concatenate`
            layer. `depth_multiplier` of the `DepthwiseConv2D` layer is also defined as
            `ratio-1`.
        use_se (bool): Determine whether or not use a squeeze-and-excitation module before
            the last `ghost_module` layer.
        name (str): Name of the module.
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.
    Returns:
        Output feature map, size = $(H,W,\mathrm{out})$.
    """

    def __init__(
        self,
        dwkernel: int,
        strides: int,
        exp: int,
        out: int,
        ratio: int,
        use_se: bool,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dwkernel = dwkernel
        self.strides = strides
        self.exp = exp
        self.out = out
        self.ratio = ratio
        self.use_se = use_se
        self.l2_regul = l2_regul

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()
        self.bn5 = BatchNormalization()
        self.relu = ReLU()
        self.add = Add()

    def build(self, input_shape) -> None:

        dwconv_config = {
            "padding": "same",
            "activation": None,
            "use_bias": False,
            "depthwise_initializer": "he_uniform",
            "depthwise_regularizer": tf.keras.regularizers.l2(l2=self.l2_regul),
        }

        self.dwconv1 = DepthwiseConv2D(
            kernel_size=self.dwkernel,
            strides=self.strides,
            depth_multiplier=self.ratio - 1,
            **dwconv_config,
        )

        self.dwconv2 = DepthwiseConv2D(
            kernel_size=self.dwkernel,
            strides=self.strides,
            depth_multiplier=self.ratio - 1,
            **dwconv_config,
        )

        self.conv = Conv2D(
            filters=self.out,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation=None,
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.ghost1 = GhostModule(
            out=self.exp,
            ratio=self.ratio,
            convkernel=1,
            dwkernel=3,
        )

        self.ghost2 = GhostModule(
            out=self.out,
            ratio=self.ratio,
            convkernel=1,
            dwkernel=3,
        )

        self.se = SqueezeAndExcite(
            filters=self.exp,
            ratio=self.ratio,
        )

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        fmap1 = self.dwconv1(inputs)
        fmap1 = self.bn1(fmap1)
        fmap1 = self.conv(fmap1)
        fmap1 = self.bn2(fmap1)

        fmap2 = self.ghost1(inputs)
        fmap2 = self.bn3(fmap2)
        fmap2 = self.relu(fmap2)

        if self.strides > 1:
            fmap2 = self.dwconv2(fmap2)
            fmap2 = self.bn4(fmap2)
        if self.use_se:
            fmap2 = self.se(fmap2)

        fmap2 = self.ghost2(fmap2)
        fmap2 = self.bn5(fmap2)

        return self.add([fmap1, fmap2])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dwkernel": self.dwkernel,
                "strides": self.strides,
                "exp": self.exp,
                "out": self.out,
                "ratio": self.ratio,
                "use_se": self.use_se,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GhostNet(TFModel):
    def __init__(self, img_shape: List[int]) -> None:
        self.img_shape = img_shape

        self.endpoint_layers = [
            "GhostBottleneckModule_2",
            "GhostBottleneckModule_4",
            "GhostBottleneckModule_10",
            "last_conv",
        ]

    def get_classification_backbone(self) -> Model:
        dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
        exps = [
            16,
            48,
            72,
            72,
            120,
            240,
            200,
            184,
            184,
            480,
            672,
            672,
            960,
            960,
            960,
            960,
        ]
        outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
        ratios = [2] * 16
        use_ses = [
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
        ]

        params = zip(dwkernels, strides, exps, outs, ratios, use_ses)

        modules = [
            GhostBottleneckModule(
                dwkernel=param[0],
                strides=param[1],
                exp=param[2],
                out=param[3],
                ratio=param[4],
                use_se=param[5],
                name=f"GhostBottleneckModule_{idx}",
            )
            for idx, param in enumerate(params)
        ]

        return Sequential(
            [
                Input(self.img_shape),
                Conv2D(
                    filters=16,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                ),
                *modules,
                Conv2D(
                    filters=960,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    name="last_conv",
                ),
            ],
            name="GhostNet",
        )

    def get_segmentation_backbone(self) -> Model:
        backbone = self.get_classification_backbone()

        os4_output, os8_output, os16_output, os32_output = [
            backbone.get_layer(layer_name).output for layer_name in self.endpoint_layers
        ]

        return Model(
            inputs=[backbone.input],
            outputs=[os4_output, os8_output, os16_output, os32_output],
            name="GhostNet",
        )
