from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

from db import connect_to_db, insert, search
from model import infer_embedding

# Load environment variables
load_dotenv()

# Init FastAPI app
app = FastAPI()

# PyMilvus connection
connect_to_db()


@app.post("/register/")
async def register_image(image_id: str, file: UploadFile = File(...), model_id: str = "resnet50"):
    try:
        image_bytes = await file.read()

        insert(
            image_id,
            model_id,
            infer_embedding(image_bytes, model_id),
        )
    except Exception as e:
        return e
    else:
        return {"message": "Image registered", "Image ID": image_id}


@app.post("/retrieve/")
async def retrieve_similar_images(image_id: str, model_id: str = "resnet50", topk: int = 10):
    response = search(image_id, model_id, topk)
    return response
