## Lava-DL SLAYER Object detection module

Lava-dl now includes object detection module which can be accessed as `from lava.lib.dl.slayer import obd` or as `slayer.obd`. The object detection module includes

* Base YOLO class `obd.yolo_base` which can be used to design and train YOLO SNNs.
* Pre-formulated model descriptions and pre-trained models `obd.models.{tiny_yolov3_str, yolo_kp, *}`.
* Bounding box metrics and utilities `obd.bbox.{metrics, utils}` to facilitate video object detection training with spiking neurons.
* Dataset modules and utilities `obd.dataset.{BDD, utils}`. Currently there is support for [Berkley Deep Drive (BDD100K)](https://bdd-data.berkeley.edu/) dataset. More dataset support will be added in the future.