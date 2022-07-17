from abc import ABC, abstractmethod
from typing import Dict

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class BaseModel(ABC):
    @abstractmethod
    def test_model_constructor(self, model: Model):
        assert isinstance(model, Model)

    @abstractmethod
    def test_backbone(self, fmap):
        pass

    @abstractmethod
    def test_classification_model(self, fmap):
        pass

    @abstractmethod
    def test_segmentation_model(self, fmap):
        pass


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
