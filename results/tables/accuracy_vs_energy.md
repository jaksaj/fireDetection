# Accuracy per millijoule, cheapest power mode

| method     | label                 | best_mode   |   energy_mj |   latency_ms |   f1_macro |   mj_per_f1_point |
|:-----------|:----------------------|:------------|------------:|-------------:|-----------:|------------------:|
| iteration1 | FireCNN (binary cls)  | 15W         |      6.1930 |       0.6046 |     0.9169 |            6.7546 |
| iteration2 | MobileNetV3-S (4-cls) | 25W         |      8.1060 |       0.9756 |     0.9411 |            8.6134 |
| iteration3 | MobileNetV3-S robust  | 25W         |      8.1470 |       0.9812 |     0.9510 |            8.5666 |
| iteration5 | U-Net (segmentation)  | 15W         |     75.2500 |       6.5068 |     0.6999 |          107.5215 |
| iteration4 | YOLO26n (detection)   | 15W         |     82.8980 |       9.0345 |     0.9689 |           85.5579 |
