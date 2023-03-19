from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("../models/1")
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

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
    return confidence, img_class

if __name__ =="__main__":
    uvicorn.run(app, host="localhost", port=8000)