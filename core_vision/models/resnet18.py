from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Layer, ReLU
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable()
class ResNetBlock(Layer):
    def __init__(
        self,
        filters: int,
        downsample: bool = False,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.downsample = downsample
        self.l2_regul = l2_regul

        self.act = ReLU()

        if downsample:
            self.strides = 2
        else:
            self.strides = 1

    def build(self, input_shape) -> None:

        self.conv1 = Conv2D(
            filters=self.filters,
            strides=self.strides,
            kernel_size=3,
            padding="same",
            use_bias=True,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=self.filters,
            strides=1,
            kernel_size=3,
            padding="same",
            use_bias=True,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.bn2 = BatchNormalization()

        if self.downsample:
            self.res_conv = Conv2D(
                self.filters,
                strides=2,
                kernel_size=1,
                kernel_initializer="he_uniform",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            )
            self.res_bn = BatchNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        skip = inputs

        fmap = self.conv1(inputs)
        fmap = self.bn1(fmap)
        fmap = self.act(fmap)
        fmap = self.conv2(fmap)
        fmap = self.bn2(fmap)

        if self.downsample:
            skip = self.res_conv(skip)
            skip = self.res_bn(skip)

        fmap = Add()([fmap, skip])
        return self.act(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"filters": self.filters, "downsample": self.downsample},
        )
        return config

    @classmethod
    def from_config(cls, config) -> None:
        return cls(**config)


class ResNet18(Model):
    def __init__(self, *args, **kwargs):
        """
        num_classes: number of classes in specific classification task.
        """
        super().__init__(*args, **kwargs)
        # self.image_shape = image_shape

        self.conv1 = Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
        )

        self.init_bn = BatchNormalization()
        # self.pool = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")

        self.res11 = ResNetBlock(
            filters=64,
            downsample=True,
            name="resnet_block1_layer1",
        )
        self.res12 = ResNetBlock(filters=64, name="resnet_block1_layer2")

        self.res21 = ResNetBlock(
            filters=128,
            downsample=True,
            name="resnet_block2_layer1",
        )
        self.res22 = ResNetBlock(filters=128, name="resnet_block2_layer2")

        self.res31 = ResNetBlock(
            filters=256,
            downsample=True,
            name="resnet_block3_layer1",
        )
        self.res32 = ResNetBlock(filters=256, name="resnet_block3_layer2")

        self.res41 = ResNetBlock(
            filters=512,
            downsample=True,
            name="resnet_block4_layer1",
        )
        self.res42 = ResNetBlock(filters=512, name="resnet_block4_layer2")

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        # out = self.pool(out)
        for res_block in [
            self.res11,
            self.res12,
            self.res21,
            self.res22,
            self.res31,
            self.res32,
            self.res41,
            self.res42,
        ]:
            out = res_block(out)

        return out
