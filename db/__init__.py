import os

from fastapi import HTTPException
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)

from model import get_embedding_dimensions
from util.constants import DB_EMBEDDING_COL, DB_IMAGE_ID_COL, INDEX_PARAM


def connect_to_db(host: str = os.environ.get("DB_HOST"), port: str = os.environ.get("DB_PORT")):
    connections.connect(host=host, port=port)


def get_collection(model_id: str):
    collection_name = model_id
    if not utility.has_collection(collection_name):
        schema = CollectionSchema(
            fields=[
                FieldSchema(name=DB_EMBEDDING_COL, dtype=DataType.FLOAT_VECTOR, dim=get_embedding_dimensions(model_id)),
                FieldSchema(name=DB_IMAGE_ID_COL, dtype=DataType.VARCHAR, max_length=255, is_primary=True),
            ]
        )
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name=DB_EMBEDDING_COL, index_params=INDEX_PARAM)
        collection.load()
        return collection
    else:
        collection = Collection(collection_name)
        collection.load()
        return collection


def insert(image_id: str, model_id: str, embedding: list):
    collection = get_collection(model_id)
    collection.insert([[embedding], [image_id]])
    collection.flush()


def search(image_id: str, model_id: str, topk: int):
    collection = get_collection(model_id)

    # Retrieve the embedding for the given image_id
    query = f"image_id == '{image_id}'"
    stored_embeddings = collection.query(query, output_fields=[DB_EMBEDDING_COL])

    if not stored_embeddings:
        raise HTTPException(status_code=404, detail="Image ID not found.")

    embedding_to_search = stored_embeddings[0][DB_EMBEDDING_COL]
    results = collection.search(anns_field=DB_EMBEDDING_COL, data=embedding_to_search, limit=topk, param=INDEX_PARAM)

    return {"image_ids": [result.entity.get(DB_IMAGE_ID_COL) for result in results[0]]}
