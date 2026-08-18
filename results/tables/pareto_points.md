# Accuracy vs cost, per method and device

| method     | label                 | device_class           | device                 | backend           | precision   |   latency_ms |       fps |   f1_macro |   accuracy |
|:-----------|:----------------------|:-----------------------|:-----------------------|:------------------|:------------|-------------:|----------:|-----------:|-----------:|
| iteration1 | FireCNN (binary cls)  | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]  | fp32        |       2.3401 |  427.3230 |     0.9169 |     0.9380 |
| iteration1 | FireCNN (binary cls)  | Desktop GPU (RTX 3060) | cuda                   | pytorch           | fp32        |       0.7171 | 1394.5057 |     0.9169 |     0.9380 |
| iteration1 | FireCNN (binary cls)  | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]  | fp32        |      12.0127 |   83.2453 |     0.9169 |     0.9380 |
| iteration1 | FireCNN (binary cls)  | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec] | fp16        |       0.3709 | 2696.5080 |     0.9169 |     0.9380 |
| iteration2 | MobileNetV3-S (4-cls) | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]  | fp32        |       0.9616 | 1039.9334 |     0.9411 |     0.9538 |
| iteration2 | MobileNetV3-S (4-cls) | Desktop GPU (RTX 3060) | cuda                   | pytorch           | fp32        |       5.6924 |  175.6728 |     0.9411 |     0.9538 |
| iteration2 | MobileNetV3-S (4-cls) | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]  | fp32        |       4.5440 |  220.0720 |     0.9411 |     0.9538 |
| iteration2 | MobileNetV3-S (4-cls) | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec] | fp16        |       0.9040 | 1106.2045 |     0.9411 |     0.9538 |
| iteration3 | MobileNetV3-S robust  | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]  | fp32        |       1.0509 |  951.5201 |     0.9510 |     0.9624 |
| iteration3 | MobileNetV3-S robust  | Desktop GPU (RTX 3060) | cuda                   | pytorch           | fp32        |       5.8132 |  172.0238 |     0.9510 |     0.9624 |
| iteration3 | MobileNetV3-S robust  | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]  | fp32        |       4.5652 |  219.0492 |     0.9510 |     0.9624 |
| iteration3 | MobileNetV3-S robust  | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec] | fp16        |       0.9163 | 1091.3933 |     0.9510 |     0.9624 |
| iteration4 | YOLO26n (detection)   | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]  | fp32        |      22.7728 |   43.9119 |     0.9689 |     0.9758 |
| iteration4 | YOLO26n (detection)   | Desktop GPU (RTX 3060) | cuda                   | pytorch           | fp32        |      16.6199 |   60.1690 |     0.9689 |     0.9758 |
| iteration4 | YOLO26n (detection)   | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]  | fp32        |      77.7567 |   12.8606 |     0.9689 |     0.9758 |
| iteration4 | YOLO26n (detection)   | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec] | fp16        |       5.5486 |  180.2263 |     0.9689 |     0.9758 |
| iteration5 | U-Net (segmentation)  | Desktop x86 CPU        | cpu                    | onnxruntime[CPU]  | fp32        |      69.4313 |   14.4027 |     0.6999 |     0.7559 |
| iteration5 | U-Net (segmentation)  | Desktop GPU (RTX 3060) | cuda                   | pytorch           | fp16        |       3.8975 |  256.5714 |     0.6999 |     0.7559 |
| iteration5 | U-Net (segmentation)  | Jetson ARM CPU         | jetson-cpu@MAXN_SUPER  | onnxruntime[CPU]  | fp32        |     230.8251 |    4.3323 |     0.6999 |     0.7559 |
| iteration5 | U-Net (segmentation)  | Jetson GPU (TensorRT)  | jetson-cuda@MAXN_SUPER | tensorrt[trtexec] | fp16        |       3.9573 |  252.6988 |     0.6999 |     0.7559 |
