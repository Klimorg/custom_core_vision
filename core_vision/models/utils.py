from abc import ABC, abstractmethod

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class TFModel(ABC):
    @abstractmethod
    def get_classification_backbone(
        self,
    ) -> Model:
        """Instantiate the model and use it as a backbone (feature extractor) for a classification task.

        Returns:
            A `tf.keras` model.
        """
        pass

    @abstractmethod
    def get_segmentation_backbone(
        self,
    ) -> Model:
        """Instantiate the model and use it as a backbone (feature extractor) for a semantic segmentation task.

        Returns:
            A `tf.keras` model.
        """
        pass
