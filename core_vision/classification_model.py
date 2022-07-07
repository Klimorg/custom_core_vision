from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class ClassificationModel(Model):
    def __init__(self, backbone: Model, classification_head: Layer, *args, **kwargs):
        """_summary_

        Args:
            backbone (Model): _description_
            classification_head (Layer): _description_
        """
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.classification_head = classification_head

    def call(self, inputs):
        fmap = self.backbone(inputs)

        return self.classification_head(fmap)
