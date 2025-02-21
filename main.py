from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import onnxruntime as ort

app = FastAPI()

# Load model
model_path = "C:/Users/ADMIN/model_repository/densenet_onnx/1/model.onnx"
session = ort.InferenceSession(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)): 
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image file"}

    # Tiền xử lý ảnh
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Định dạng (3, 224, 224)
    image = np.expand_dims(image, axis=0)   # Định dạng (1, 3, 224, 224)

    # Chạy inference
    input_name = "data_0"
    output_name = "fc6_1"
    outputs = session.run([output_name], {input_name: image})

    predictions = outputs[0].reshape(1, 1000).tolist()

    return {"filename": file.filename, "prediction": predictions}

