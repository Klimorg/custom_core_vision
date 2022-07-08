import numpy as np

# from hypothesis import given
# from hypothesis import strategies as st
# from hypothesis.extra.numpy import arrays
from pytest import fixture
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.classification_model import ClassificationModel
from core_vision.heads.classification_head import ClassificationHead
from core_vision.models.resnet18 import ResNet18, ResNetBlock
from tests.utils import BaseModel


@fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


def test_layer_constructor():

    layer1 = ResNetBlock(filters=32, downsample=False)
    layer2 = ResNetBlock(filters=32, downsample=True)

    assert isinstance(layer1, Layer)
    assert isinstance(layer2, Layer)


class TestResNet18(BaseModel):
    def __init__(self):
        self.model = ResNet18()

    def test_model_constructor(self):

        assert isinstance(self.model, Model)

    # @given(
    #     arrays(
    #         dtype=np.float32,
    #         elements=st.floats(0, 1, width=32),
    #         shape=[1, 224, 224, 3],
    #     ),
    # )
    def test_backbone(self, fmap):

        out = self.model(fmap)

        assert out.shape.as_list() == [1, 7, 7, 512]

    def test_classification_model(self, fmap):

        head = ClassificationHead(units=32, num_classes=10)

        model = ClassificationModel(
            backbone=self.model,
            classification_head=head,
            name=f"{self.model.name}_classification",
        )

        out = model(fmap)

        assert isinstance(model, Model)
        assert out.shape.as_list()[1] == 10

    def test_segmentation_model(self):
        pass
