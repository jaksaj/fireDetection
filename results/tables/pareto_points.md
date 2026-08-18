# Accuracy vs cost, per method and device

| method     | label                 | device_class           | device                 | backend              | precision   |   latency_ms |       fps |   f1_macro |   accuracy |
|:-----------|:----------------------|:-----------------------|:-----------------------|:---------------------|:------------|-------------:|----------:|-----------:|-----------:|
| iteration1 | FireCNN (binary cls)  | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]     | int8-static |       1.9930 |  501.7561 |     0.9169 |     0.9380 |
| iteration1 | FireCNN (binary cls)  | Desktop GPU (RTX 3060) | cuda                   | tensorrt[python-api] | fp16        |       0.1289 | 7757.9519 |     0.9169 |     0.9380 |
| iteration1 | FireCNN (binary cls)  | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]     | fp32        |      12.0127 |   83.2453 |     0.9169 |     0.9380 |
| iteration1 | FireCNN (binary cls)  | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec]    | fp16        |       0.3709 | 2696.5080 |     0.9169 |     0.9380 |
| iteration2 | MobileNetV3-S (4-cls) | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]     | fp32        |       0.9616 | 1039.9334 |     0.9411 |     0.9538 |
| iteration2 | MobileNetV3-S (4-cls) | Desktop GPU (RTX 3060) | cuda                   | tensorrt[python-api] | fp16        |       0.6589 | 1517.5658 |     0.9411 |     0.9538 |
| iteration2 | MobileNetV3-S (4-cls) | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]     | fp32        |       4.5440 |  220.0720 |     0.9411 |     0.9538 |
| iteration2 | MobileNetV3-S (4-cls) | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec]    | fp16        |       0.9040 | 1106.2045 |     0.9411 |     0.9538 |
| iteration3 | MobileNetV3-S robust  | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]     | fp32        |       1.0509 |  951.5201 |     0.9510 |     0.9624 |
| iteration3 | MobileNetV3-S robust  | Desktop GPU (RTX 3060) | cuda                   | tensorrt[python-api] | fp16        |       0.6700 | 1492.5373 |     0.9510 |     0.9624 |
| iteration3 | MobileNetV3-S robust  | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]     | fp32        |       4.5652 |  219.0492 |     0.9510 |     0.9624 |
| iteration3 | MobileNetV3-S robust  | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec]    | fp16        |       0.9163 | 1091.3933 |     0.9510 |     0.9624 |
| iteration4 | YOLO26n (detection)   | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]     | int8-static |      16.4606 |   60.7509 |     0.9689 |     0.9758 |
| iteration4 | YOLO26n (detection)   | Desktop GPU (RTX 3060) | cuda                   | tensorrt[python-api] | fp16        |       2.2574 |  442.9973 |     0.9689 |     0.9758 |
| iteration4 | YOLO26n (detection)   | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]     | fp32        |      77.7567 |   12.8606 |     0.9689 |     0.9758 |
| iteration4 | YOLO26n (detection)   | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec]    | fp16        |       5.5486 |  180.2263 |     0.9689 |     0.9758 |
| iteration5 | U-Net (segmentation)  | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]     | int8-static |      32.2802 |   30.9787 |     0.6999 |     0.7559 |
| iteration5 | U-Net (segmentation)  | Desktop GPU (RTX 3060) | cuda                   | tensorrt[python-api] | fp16        |       1.2175 |  821.3552 |     0.6999 |     0.7559 |
| iteration5 | U-Net (segmentation)  | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]     | fp32        |     230.8251 |    4.3323 |     0.6999 |     0.7559 |
| iteration5 | U-Net (segmentation)  | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec]    | fp16        |       3.9573 |  252.6988 |     0.6999 |     0.7559 |
