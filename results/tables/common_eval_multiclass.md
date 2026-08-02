# Common-task comparison (multiclass axis), best operating point per method

| method     | model_name               | paradigm                  |   threshold |   n_images |   accuracy |   f1_macro |   f1_fire |   precision_fire |   recall_fire |   f1_Neither |   f1_Only_Fire |   f1_Only_Smoke |   f1_Both |   domain_shift |
|:-----------|:-------------------------|:--------------------------|------------:|-----------:|-----------:|-----------:|----------:|-----------------:|--------------:|-------------:|---------------:|----------------:|----------:|---------------:|
| iteration4 | YOLO26n                  | object detection          |      0.1000 |       4306 |     0.9366 |     0.8944 |       nan |              nan |           nan |       0.9719 |         0.7670 |          0.9235 |    0.9151 |              0 |
| iteration3 | MobileNetV3-Small robust | multiclass classification |    nan      |       4306 |     0.8955 |     0.8610 |       nan |              nan |           nan |       0.9408 |         0.7680 |          0.8620 |    0.8734 |              0 |
| iteration2 | MobileNetV3-Small        | multiclass classification |    nan      |       4306 |     0.8946 |     0.8438 |       nan |              nan |           nan |       0.9483 |         0.6878 |          0.8804 |    0.8586 |              0 |
| iteration5 | LightweightUNet          | semantic segmentation     |      0.0200 |       4306 |     0.4445 |     0.3669 |       nan |              nan |           nan |       0.5263 |         0.0388 |          0.3264 |    0.5761 |              1 |
