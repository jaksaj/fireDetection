# RTX 3060: FP32 vs FP16 under eager PyTorch and TensorRT (batch 1)

| model      | label                 |   eager_fp32_ms |   eager_fp16_ms |   eager_fp16_speedup |   trt_fp32_ms |   trt_fp16_ms |   trt_fp16_speedup |
|:-----------|:----------------------|----------------:|----------------:|---------------------:|--------------:|--------------:|-------------------:|
| iteration1 | FireCNN (binary cls)  |          0.7171 |          0.8278 |               0.8662 |        0.2794 |        0.1289 |             2.1676 |
| iteration2 | MobileNetV3-S (4-cls) |          5.6924 |          6.4817 |               0.8782 |        0.7568 |        0.6589 |             1.1484 |
| iteration3 | MobileNetV3-S robust  |          5.8132 |          6.5127 |               0.8926 |        0.7184 |        0.6700 |             1.0722 |
| iteration4 | YOLO26n (detection)   |         16.6199 |         19.5128 |               0.8517 |        3.2219 |        2.2574 |             1.4273 |
| iteration5 | U-Net (segmentation)  |          4.8795 |          3.8975 |               1.2519 |        3.1449 |        1.2175 |             2.5830 |
