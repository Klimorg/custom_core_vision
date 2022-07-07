import numpy as np
from pytest import fixture
from tensorflow.keras.models import Model

from core_vision.classification_model import ClassificationModel
from core_vision.heads.classification_head import ClassificationHead
from core_vision.models.resnet18 import ResNet18


@fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)


def compute_model(fmap):
    backbone = ResNet18()
    head = ClassificationHead(units=32, num_classes=10)

    model = ClassificationModel(backbone=backbone, classification_head=head)

    out = model(fmap)

    assert isinstance(model, Model)
    assert out.shape.as_list()[1] == 10
