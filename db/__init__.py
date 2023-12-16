import pymilvus
from fastapi import HTTPException

from db.model import SearchParam
from model import get_embedding_dimensions


def get_collection(model_id: str):
    collection_name = model_id
    if not pymilvus.schema.CollectionSchema.exists(collection_name):
        schema = pymilvus.schema.CollectionSchema(fields=[
            pymilvus.schema.FieldSchema(
                name="embedding", 
                dtype=pymilvus.DataType.FLOAT_VECTOR,
                dim=get_embedding_dimensions(model_id)
            ),
            pymilvus.schema.FieldSchema(
                name="image_id",
                dtype=pymilvus.DataType.VARCHAR,
                max_length=255,
                is_primary=True
            )
        ])
        return pymilvus.Collection(name=collection_name, schema=schema)
    else:
        return pymilvus.Collection(collection_name)
        

def insert(image_id: str, model_id: str, embedding: list):
    collection = get_collection(model_id)    
    collection.insert([[embedding], [image_id]])


def search(image_id: str, model_id: str, search_params: SearchParam):
    collection = get_collection(model_id)

    # Retrieve the embedding for the given image_id
    search_param = {"metric_type": search_params.metric_type, "params": {"nprobe": search_params.nprobe}}
    query = f"image_id == '{image_id}'"
    stored_embeddings = collection.query(query, output_fields=["embedding"])

    if not stored_embeddings:
        raise HTTPException(status_code=404, detail="Image ID not found.")

    # Assuming the first returned embedding is the one we want
    embedding_to_search = stored_embeddings[0]["embedding"]

    # Search in PyMilvus
    results = collection.search([embedding_to_search], "embedding", search_param, limit=10)

    # Extract image IDs from results
    similar_image_ids = [result.entity.get("image_id") for result in results[0]]

    return {"similar_image_ids": similar_image_ids}
