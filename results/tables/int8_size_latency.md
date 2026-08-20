# Static INT8: artifact size and x86 CPU latency

| model      | label                 |   fp32_onnx_mb |   int8_onnx_mb |   compression |   x86_fp32_ms |   x86_int8_ms |   x86_speedup | artifact             |
|:-----------|:----------------------|---------------:|---------------:|--------------:|--------------:|--------------:|--------------:|:---------------------|
| iteration1 | FireCNN (binary cls)  |         1.4860 |         0.3921 |        3.7900 |        2.3401 |        1.9930 |        1.1742 | iteration1_int8.onnx |
| iteration2 | MobileNetV3-S (4-cls) |         4.1220 |         1.3743 |        2.9994 |        0.9616 |        1.3355 |        0.7200 | iteration2_int8.onnx |
| iteration3 | MobileNetV3-S robust  |         4.1220 |         1.3743 |        2.9994 |        1.0509 |        1.1880 |        0.8846 | iteration3_int8.onnx |
| iteration4 | YOLO26n (detection)   |         9.6608 |         3.1104 |        3.1060 |       22.7728 |       16.4606 |        1.3835 | iteration4_int8.onnx |
| iteration5 | U-Net (segmentation)  |        29.9446 |         7.5733 |        3.9540 |       69.4313 |       32.2802 |        2.1509 | iteration5_int8.onnx |
