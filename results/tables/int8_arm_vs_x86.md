# INT8 speedup on ARM (Cortex-A78AE) vs x86 (Ryzen), same ONNX and runtime

| model      | label                 |   arm_fp32_ms |   arm_int8_ms |   arm_speedup |   x86_fp32_ms |   x86_int8_ms |   x86_speedup |
|:-----------|:----------------------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| iteration1 | FireCNN (binary cls)  |       15.1880 |        5.7440 |        2.6442 |        2.3401 |        1.9930 |        1.1742 |
| iteration2 | MobileNetV3-S (4-cls) |        5.5930 |        4.8540 |        1.1522 |        0.9616 |        1.3355 |        0.7200 |
| iteration3 | MobileNetV3-S robust  |        5.6200 |        4.8450 |        1.1600 |        1.0509 |        1.1880 |        0.8846 |
| iteration4 | YOLO26n (detection)   |       96.1350 |       51.5530 |        1.8648 |       22.7728 |       16.4606 |        1.3835 |
| iteration5 | U-Net (segmentation)  |      291.2970 |       90.2650 |        3.2271 |       69.4313 |       32.2802 |        2.1509 |
