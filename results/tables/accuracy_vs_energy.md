# Accuracy per millijoule, cheapest power mode

| method     | label                 | best_mode   |   energy_mj |   latency_ms |   f1_macro |   mj_per_f1_point |
|:-----------|:----------------------|:------------|------------:|-------------:|-----------:|------------------:|
| iteration1 | FireCNN (binary cls)  | 15W         |      6.1930 |       0.6046 |     0.9036 |            6.8536 |
| iteration2 | MobileNetV3-S (4-cls) | 25W         |      8.1060 |       0.9756 |     0.9445 |            8.5820 |
| iteration3 | MobileNetV3-S robust  | 25W         |      8.1470 |       0.9812 |     0.9477 |            8.5964 |
| iteration5 | U-Net (segmentation)  | 15W         |     75.2500 |       6.5068 |     0.7117 |          105.7261 |
| iteration4 | YOLO26n (detection)   | 15W         |     82.8980 |       9.0345 |     0.9692 |           85.5368 |
