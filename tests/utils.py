from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def test_model_constructor(self):
        pass

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
    def test_layer_constructor(self):
        pass

    @abstractmethod
    def test_layer(self, fmap):
        pass

    @abstractmethod
    def test_config(self):
        pass
