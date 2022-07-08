from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def test_model_constructor(self):
        pass

    @abstractmethod
    def test_backbone(self):
        pass

    @abstractmethod
    def test_classification_model(self):
        pass

    @abstractmethod
    def test_segmentation_model(self):
        pass


class BaseLayer(ABC):
    @abstractmethod
    def test_layer_constructor(self):
        pass

    @abstractmethod
    def test_layer(self):
        pass

    @abstractmethod
    def test_config(self):
        pass
