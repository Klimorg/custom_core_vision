from typing import List

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class SegmentationModel(Model):
    def __init__(
        self,
        backbone: Model,
        segmentation_head: Layer,
        endpoint_layers: List[str],
        name: str,
        *args,
        **kwargs,
    ):

        super().__init__(name=name, *args, **kwargs)
        self.backbone = backbone
        self.segmentation_head = segmentation_head
        self.endpoint_layers = endpoint_layers
        self.segmentation_backbone = self.get_backbone_endpoints()

    def get_backbone_endpoints(self) -> Model:

        # endpoint_layers = [
        #     "convnext_layer_1",
        #     "convnext_layer_2",
        #     "convnext_layer_3",
        #     "convnext_layer_4",
        # ]

        os4_output, os8_output, os16_output, os32_output = [
            self.backbone.get_layer(layer_name).output
            for layer_name in self.endpoint_layers
        ]

        # height = img_shape[1]
        # logger.info(f"os4_output OS : {int(height/os4_output.shape.as_list()[1])}")
        # logger.info(f"os8_output OS : {int(height/os8_output.shape.as_list()[1])}")
        # logger.info(f"os16_output OS : {int(height/os16_output.shape.as_list()[1])}")
        # logger.info(f"os32_output OS : {int(height/os32_output.shape.as_list()[1])}")

        return Model(
            inputs=[self.backbone.input],
            outputs=[os4_output, os8_output, os16_output, os32_output],
            name=self.backbone.name,
        )

    def call(self, inputs):
        fmap = self.segmentation_backbone(inputs)

        return self.segmentation_head(fmap)
