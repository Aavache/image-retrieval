import os

import pymilvus
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

from db import insert, search
from db.model import SearchParam
from model import infer_embedding

# Load environment variables
load_dotenv()

# Init FastAPI app
app = FastAPI()

# PyMilvus connection
pymilvus.connections.connect(os.environ.get("DB_NAME"), host=os.environ.get("DB_HOST"), port=os.environ.get("DB_PORT"))


def register(image_id: str, file: UploadFile = File(...), model_id: str = "resnet50"):
    try:
        image_bytes = file.read()

        insert(
            image_id,
            model_id,
            infer_embedding(image_bytes, model_id),
        )
    except Exception as e:
        return e
    else:
        return {"message": "Image registered", "Image ID": image_id}


def retrieve(image_id: str, model_id: str = "resnet50", search_params: SearchParam = None):
    response = search(image_id, model_id, search_params)
    return response


def main():

if __name__ == "__main__":
    main()