# Common-task comparison (multiclass axis), best operating point per method

| method     | model_name               | paradigm                  |   threshold |   n_images |   accuracy |   f1_macro |   f1_fire |   precision_fire |   recall_fire |   domain_shift |
|:-----------|:-------------------------|:--------------------------|------------:|-----------:|-----------:|-----------:|----------:|-----------------:|--------------:|---------------:|
| iteration4 | YOLO26n                  | object detection          |      0.1000 |       4306 |     0.9364 |     0.8900 |       nan |              nan |           nan |              0 |
| iteration3 | MobileNetV3-Small robust | multiclass classification |    nan      |       4306 |     0.8908 |     0.8527 |       nan |              nan |           nan |              0 |
| iteration2 | MobileNetV3-Small        | multiclass classification |    nan      |       4306 |     0.8886 |     0.8445 |       nan |              nan |           nan |              0 |
| iteration5 | LightweightUNet          | semantic segmentation     |      0.0200 |       4306 |     0.4785 |     0.3767 |       nan |              nan |           nan |              1 |
