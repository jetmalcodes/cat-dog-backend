from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.layers import TFSMLayer
from PIL import Image
import numpy as np
import io
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use the correct model path
MODEL_PATH = "model/cat_dog_model_export"
model_layer = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0).astype("float32")

    preds = model_layer(arr)
    if isinstance(preds, dict):
        pred = list(preds.values())[0].numpy()[0][0]
    else:
        pred = preds.numpy()[0][0]

    result = "Dog" if pred > 0.5 else "Cat"
    return {"result": result}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
