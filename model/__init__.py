from typing import List

from fastapi import HTTPException

from model.resnet import ResNet50

MODEL_REGISTRY = {
    "resnet50": ResNet50,
}


def check_model_exist(model_id: str) -> None | HTTPException:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model {model_id} not supported")


def get_embedding_dimensions(model_id: str) -> int:
    check_model_exist(model_id)
    return MODEL_REGISTRY.get(model_id, None).embedding_dimensions()


def infer_embedding(image_bytes, model_id: str) -> List[float]:
    check_model_exist(model_id)
    model = MODEL_REGISTRY[model_id]()
    return model.get_embedding(image_bytes)
