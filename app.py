import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model('model.keras')
IMAGE_SIZE = 255
@app.get("/")
def home():
    return {"message": "Welcome to my API"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image)

    result = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return {
        "prediction": result,
        "confidence": float(prediction[0][0])
    }