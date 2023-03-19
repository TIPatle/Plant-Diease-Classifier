from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("../models/1")
class_names = ['Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot', 'Tomato YellowLeaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

@app.get("/ping")
async def ping():
    return "I'm alive."

@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    img = await file.read(file.size)
    image = np.array(Image.open(BytesIO(img)))
    image = np.expand_dims(image, 0)
    prediction = model.predict(image)
    confidence = np.max(prediction[0])*100
    img_class = class_names[np.argmax(prediction[0])]
    return {
        "Confidence": confidence, 
        "Class": img_class
    }

if __name__ =="__main__":
    uvicorn.run(app, host="localhost", port=5000)