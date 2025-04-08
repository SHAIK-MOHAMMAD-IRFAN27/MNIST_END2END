# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model
model = tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    predicted_label = int(np.argmax(prediction))
    return {"prediction": predicted_label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


