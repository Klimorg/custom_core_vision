import numpy as np
import pytest
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.classification_model import ClassificationModel
from core_vision.heads.classification_head import ClassificationHead
from core_vision.models.convnext import ConvNext, ConvNextBlock, ConvNeXtLayer
from tests.utils import BaseModel


@pytest.fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


def test_layer_constructor():

    layer1 = ConvNextBlock(filters=32)
    layer2 = ConvNeXtLayer(filters=32, num_blocks=4)

    assert isinstance(layer1, Layer)
    assert isinstance(layer2, Layer)


@pytest.mark.parametrize(
    "filters, num_blocks, name",
    [
        ([128, 256, 384, 768], [3, 3, 27, 3], "convnext-b"),
        ([192, 384, 768, 1536], [3, 3, 27, 3], "convnext-l"),
        ([96, 182, 384, 768], [3, 3, 27, 3], "convnext-s"),
        ([96, 182, 384, 768], [3, 3, 9, 3], "convnext-t"),
        ([256, 512, 1024, 2048], [3, 3, 27, 3], "convnext-xl"),
    ],
)
class TestConvNext(BaseModel):
    def test_model_constructor(self, filters, num_blocks, name):
        model = ConvNext(filters=filters, num_blocks=num_blocks, name=name)

        assert isinstance(model, Model)

    def test_backbone(self, fmap, filters, num_blocks, name):
        model = ConvNext(filters=filters, num_blocks=num_blocks, name=name)
        out = model(fmap)

        assert out.shape.as_list() == [1, 7, 7, filters[3]]

    def test_classification_model(self, fmap, filters, num_blocks, name):
        backbone = ConvNext(filters=filters, num_blocks=num_blocks, name=name)

        head = ClassificationHead(units=32, num_classes=10)

        model = ClassificationModel(
            backbone=backbone,
            classification_head=head,
            name=f"{backbone.name}_classification",
        )

        out = model(fmap)

        assert isinstance(model, Model)
        assert out.shape.as_list()[1] == 10

    def test_segmentation_model(self, fmap, filters, num_blocks, name):
        pass
