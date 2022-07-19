import numpy as np
import pytest
from tensorflow.keras.layers import Layer

from core_vision.layers.object_context import ASPP_OC, ISA2D, BaseOC, SelfAttention2D
from tests.utils import BaseLayer


@pytest.fixture
def fmap():
    return np.random.rand(1, 224, 224, 3)
