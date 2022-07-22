from abc import ABC, abstractmethod
from typing import Dict

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from core_vision.models.utils import TFModel


class BaseModel(ABC):
    @abstractmethod
    def test_model_constructor(self, model: Model):
        assert isinstance(model, TFModel)

    @abstractmethod
    def test_classification_backbone(self, fmap):
        pass

    @abstractmethod
    def test_segmentation_backbone(self, fmap, backbone: Model):

        out = backbone(fmap)

        assert isinstance(backbone, Model)
        assert len(out) == 4
        assert 224 / out[0].shape.as_list()[1] == 4
        assert 224 / out[1].shape.as_list()[1] == 8
        assert 224 / out[2].shape.as_list()[1] == 16
        assert 224 / out[3].shape.as_list()[1] == 32


class BaseLayer(ABC):
    @abstractmethod
    def test_layer_constructor(self, layer: Layer):
        assert isinstance(layer, Layer)

    @abstractmethod
    def test_layer(self, fmap):
        pass

    @abstractmethod
    def test_config(self, layer: Layer):
        cfg = layer.get_config()

        assert isinstance(cfg, Dict)
