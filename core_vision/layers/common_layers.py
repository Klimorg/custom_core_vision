from typing import Any, Dict

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    Layer,
    ReLU,
    SeparableConv2D,
)


@tf.keras.utils.register_keras_serializable()
class ConvGNReLU(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        padding: str = "same",
        strides: int = 1,
        dilation_rate: int = 1,
        w_init: str = "he_normal",
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            filters (int): Number of filters used in the `Conv2D` layer.
            kernel_size (int): Size of the convolution kernels used in the `Conv2D` layer.
            padding (str, optional): Padding parameter of the `Conv2D` layer. Defaults to "same".
            strides (int, optional): Strides parameter of the `Conv2D` layer. Defaults to 1.
            dilation_rate (int, optional): Dilation rate of the `Conv2D` layer. Defaults to 1.
            w_init (str, optional): Kernel initialization method used in th `Conv2D` layer. Defaults to "he_normal".
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.

        Returns:
            Output feature map, size = $(H,W,C)$.
        """
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.w_init = w_init
        self.l2_regul = l2_regul

        self.act = ReLU()

    def build(self, input_shape) -> None:

        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            kernel_initializer=self.w_init,
            use_bias=False,
        )

        self.gn = tfa.layers.GroupNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = self.conv(inputs)
        fmap = self.gn(fmap)
        return self.act(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
                "dilation_rate": self.dilation_rate,
                "w_init": self.w_init,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ConvBNReLU(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        padding: str = "same",
        strides: int = 1,
        dilation_rate: int = 1,
        w_init: str = "he_normal",
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            filters (int): Number of filters used in the `Conv2D` layer.
            kernel_size (int): Size of the convolution kernels used in the `Conv2D` layer.
            padding (str, optional): Padding parameter of the `Conv2D` layer. Defaults to "same".
            strides (int, optional): Strides parameter of the `Conv2D` layer. Defaults to 1.
            dilation_rate (int, optional): Dilation rate of the `Conv2D` layer. Defaults to 1.
            w_init (str, optional): Kernel initialization method used in th `Conv2D` layer. Defaults to "he_normal".
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.

        Returns:
            Output feature map, size = $(H,W,C)$.
        """
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.w_init = w_init
        self.l2_regul = l2_regul

        self.act = ReLU()

    def build(self, input_shape) -> None:

        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            kernel_initializer=self.w_init,
            use_bias=False,
        )

        self.bn = BatchNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = self.conv(inputs)
        fmap = self.bn(fmap)
        return self.act(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
                "dilation_rate": self.dilation_rate,
                "w_init": self.w_init,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class BNConvReLU(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        padding: str = "same",
        strides: int = 1,
        dilation_rate: int = 1,
        w_init: str = "he_normal",
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            filters (int): Number of filters used in the `Conv2D` layer.
            kernel_size (int): Size of the convolution kernels used in the `Conv2D` layer.
            padding (str, optional): Padding parameter of the `Conv2D` layer. Defaults to "same".
            strides (int, optional): Strides parameter of the `Conv2D` layer. Defaults to 1.
            dilation_rate (int, optional): Dilation rate of the `Conv2D` layer. Defaults to 1.
            w_init (str, optional): Kernel initialization method used in th `Conv2D` layer. Defaults to "he_normal".
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.

        Returns:
            Output feature map, size = $(H,W,C)$.
        """
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.w_init = w_init
        self.l2_regul = l2_regul

        self.act = ReLU()

    def build(self, input_shape) -> None:

        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            kernel_initializer=self.w_init,
            use_bias=False,
        )

        self.bn = BatchNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = self.bn(inputs)
        fmap = self.conv(fmap)
        return self.act(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
                "dilation_rate": self.dilation_rate,
                "w_init": self.w_init,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class SepConvBNReLU(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        padding: str = "same",
        strides: int = 1,
        dilation_rate: int = 1,
        w_init: str = "he_normal",
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        """_summary_

        Args:
            filters (int): Number of filters used in the `Conv2D` layer.
            kernel_size (int): Size of the convolution kernels used in the `Conv2D` layer.
            padding (str, optional): Padding parameter of the `Conv2D` layer. Defaults to "same".
            strides (int, optional): Strides parameter of the `Conv2D` layer. Defaults to 1.
            dilation_rate (int, optional): Dilation rate of the `Conv2D` layer. Defaults to 1.
            w_init (str, optional): Kernel initialization method used in th `Conv2D` layer. Defaults to "he_normal".
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.

        Returns:
            Output feature map, size = $(H,W,C)$.
        super().__init__(*args,**kwargs)
        """
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.w_init = w_init
        self.l2_regul = l2_regul

        self.act = ReLU()

    def build(self, input_shape) -> None:
        self.sepconv = SeparableConv2D(
            filters=self.filters,
            depth_multiplier=1,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            depthwise_initializer=self.w_init,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            use_bias=False,
        )

        self.bn = BatchNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = self.sepconv(inputs)
        fmap = self.bn(fmap)
        return self.act(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
                "dilation_rate": self.dilation_rate,
                "w_init": self.w_init,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class InvertedResidualBottleneck2D(Layer):
    def __init__(
        self,
        expansion_rate: int,
        filters: int,
        strides: int,
        skip_connection: bool,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.expansion_rate = expansion_rate
        self.filters = filters
        self.strides = strides
        self.skip_connection = skip_connection
        self.l2_regul = l2_regul

        if self.strides == 2:
            assert (
                self.skip_connection == False
            ), "You can't apply skip connections with strides greater than 1."

    def build(self, input_shape) -> None:

        self.conv1 = Conv2D(
            filters=self.expansion_rate * self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )
        self.conv2 = Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )

        self.act = ReLU(max_value=6)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

        self.dwconv = DepthwiseConv2D(
            kernel_size=3,
            strides=self.strides,
            padding="same",
            depth_multiplier=1,
            depthwise_initializer="he_normal",
            use_bias=False,
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = self.conv1(inputs)
        fmap = self.bn1(fmap)
        fmap = self.act(fmap)

        fmap = self.dwconv(fmap)
        fmap = self.bn2(fmap)
        fmap = self.act(fmap)

        fmap = self.conv2(fmap)
        fmap = self.bn3(fmap)

        if self.skip_connection:
            fmap += inputs

        return fmap

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "expansion_rate": self.expansion_rate,
                "filters": self.filters,
                "strides": self.strides,
                "skip_connection": self.skip_connection,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
