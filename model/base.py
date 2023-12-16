from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def embedding_dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, image_bytes: bytes) -> List[float]:
        raise NotImplementedError
