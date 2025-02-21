from fastapi import FastAPI, HTTPException, File, UploadFile
import numpy as np
import cv2
import tritonclient.http as httpclient
from pydantic import BaseModel

# Khởi tạo FastAPI
app = FastAPI(title="Triton Inference API")

# Địa chỉ của Triton Server
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "densenet_onnx"  # Thay tên model của bạn
MODEL_VERSION = "1"

# Kết nối với Triton Server
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Kiểm tra server đang chạy
@app.get("/")
async def root():
    return {"message": "Triton Inference API is running"}

# Hàm gọi Triton Server
def call_triton_server(image: np.ndarray):
    try:
        # Tiền xử lý ảnh
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # Chuyển thành (C, H, W)
        image = np.expand_dims(image, axis=0)   # Thêm batch_size = 1

        # Chuẩn bị input cho Triton
        inputs = httpclient.InferInput("data_0", image.shape, "FP32")
        inputs.set_data_from_numpy(image)

        # Chuẩn bị output từ Triton
        outputs = httpclient.InferRequestedOutput("fc6_1")

        # Gửi request đến Triton
        response = client.infer(MODEL_NAME, model_version=MODEL_VERSION, inputs=[inputs], outputs=[outputs])
        result = response.as_numpy("fc6_1")

        return result.tolist()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Triton: {str(e)}")

# API nhận file ảnh từ Postman
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ file upload
        contents = await file.read()
        print(f"File contents length: {len(contents)}")  # Log độ dài nội dung
        image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Gọi Triton Server
        predictions = call_triton_server(image)
        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))