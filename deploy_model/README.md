### Download triton server image và hosting
Setup project
```
git clone https://github.com/luuphat23/OJT
cd host_model
```

```
# Download model if model not exist
wget -O model_repository/densenet_onnx/1/model.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-7.onnx

```

Run triton server. The triton server auto install if image not exist
```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/model
```
Result
```
I0218 18:16:18.769098 1 server.cc:653] 
+---------------+---------+--------+
| Model         | Version | Status |
+---------------+---------+--------+
| densenet_onnx | 1       | READY  |
+---------------+---------+--------+
...
I0218 18:16:18.770347 1 grpc_server.cc:2450] Started GRPCInferenceService at 0.0.0.0:8001
I0218 18:16:18.770492 1 http_server.cc:3555] Started HTTPService at 0.0.0.0:8000
I0218 18:16:18.811449 1 http_server.cc:185] Started Metrics Service at 0.0.0.0:8002
```

### File config for hosting model
host_model/model_repository/densenet_onnx/config.pbtxt
```
name: "densenet"
platform: "onnxruntime_onnx"
max_batch_size: 8 
input [
  {
    name: "data" 
    data_type: TYPE_FP32
    dims: [3, 224, 224]  # Định dạng ảnh (C, H, W)
    reshape { shape: [ 1, 3, 224, 224 ] }
  }
]
output [
  {
    name: "prob" 
    data_type: TYPE_FP32
    dims: [1000]  # 1000 class output
    label_filename: "densenet_labels.txt"
  }
]
```
### Call API and run inference
```
 python client.py
```

Result
```
['34.865158:409' '31.032894:916' '30.762917:920' '30.430737:530'
 '28.804327:559']
```
### (Extra - 10đ) Đo performance với toàn bộ config có thể chỉnh sửa từ 2)  - Tutorials Link 3