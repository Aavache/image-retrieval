import io
from typing import List

import torch
from PIL import Image
from torchvision import models, transforms

from model import base


class ResNet50(base.Model):
    def __init__(self) -> None:
        self._model = models.resnet50(pretrained=True)
        self._model.eval()

    @staticmethod
    def embedding_dimensions() -> int:
        return 2048

    def _setup_transform(self) -> None:
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _preprocess(self, image_bytes) -> List[float]:
        image = Image.open(io.BytesIO(image_bytes))
        return self._transform(image).unsqueeze(0)

    def _postprocess(self, embedding: torch.Tensor) -> List[float]:
        return embedding.numpy().flatten().tolist()

    def get_embedding(self, image_bytes: bytes) -> List[float]:
        image = self._preprocess(image_bytes)
        with torch.no_grad():
            embedding = self._model(image)
        return self._postprocess(embedding)
