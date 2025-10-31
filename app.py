from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from keras.layers import TFSMLayer   # NEW
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load exported SavedModel as a callable layer
MODEL_PATH = "../model/cat_dog_model_export"
model_layer = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read + preprocess image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0).astype("float32")

    # run inference
    preds = model_layer(arr)
    # `preds` is a dict of outputs; extract first tensor
    if isinstance(preds, dict):
        pred = list(preds.values())[0].numpy()[0][0]
    else:
        pred = preds.numpy()[0][0]

    result = "Dog" if pred > 0.5 else "Cat"
    return {"result": result}
